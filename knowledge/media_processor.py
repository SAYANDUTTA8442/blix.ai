"""
Media Processor — Blix v0.3.2  (Feature 7)

Allows Blix to understand media beyond text.

Three sub-processors, sharing a common ``ProcessedMedia`` result type:

* ``ImageProcessor``   — fully functional: OCR via pytesseract + Pillow,
                         optional LLM-based scene/diagram description.
* ``AudioProcessor``   — pluggable transcription backend (interface +
                         offline stub). Plug in `openai-whisper`,
                         `faster-whisper`, or a remote ASR API by
                         implementing ``TranscriptionBackend``.
* ``VideoProcessor``   — frame sampling (via Pillow/ffmpeg if available)
                         + delegates audio track to ``AudioProcessor`` and
                         frames to ``ImageProcessor``.

All three produce the same downstream shape so they can be merged into
memory via ``MediaProcessor.to_processed_document()`` (compatible with
``knowledge.document_processor.ProcessedDocument``).

Python 3.10 compatible. Hard dependencies: Pillow, pytesseract (OCR).
Audio/video transcription backends are optional — see
``TranscriptionBackend`` for the extension point.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from llm.base import LLMProvider
from knowledge.document_processor import ProcessedDocument, DocumentFormat, DocumentChunk
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------


class MediaType(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class ProcessedMedia:
    """
    Result of processing one media file.

    Fields
    ------
    media_id:
        Stable id (filename-derived + timestamp).
    media_type:
        image / audio / video.
    title:
        Filename stem.
    summary:
        High-level natural-language summary.
    transcript:
        Full transcript (audio/video) or empty for images.
    ocr_text:
        Raw OCR text (images/video frames).
    objects:
        Detected objects/components (heuristic or LLM-described).
    diagram_notes:
        Architecture/diagram-specific notes (images).
    topics:
        Topic labels for clustering/graph.
    action_items:
        Extracted action items (audio/video).
    key_facts:
        Extracted key facts.
    segments:
        Time-coded transcript segments [(start_s, end_s, text)] (audio/video).
    chunks:
        Text chunks for embedding/indexing (compatible with DocumentChunk).
    """

    media_id: str
    media_type: MediaType
    title: str
    summary: str = ""
    transcript: str = ""
    ocr_text: str = ""
    objects: list[str] = field(default_factory=list)
    diagram_notes: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    segments: list[tuple[float, float, str]] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    processed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "media_id": self.media_id,
            "media_type": self.media_type.value,
            "title": self.title,
            "summary": self.summary,
            "transcript": self.transcript,
            "ocr_text": self.ocr_text,
            "objects": self.objects,
            "diagram_notes": self.diagram_notes,
            "topics": self.topics,
            "action_items": self.action_items,
            "key_facts": self.key_facts,
            "segments": [list(s) for s in self.segments],
            "chunks": [c.to_dict() for c in self.chunks],
            "processed_at": self.processed_at,
        }

    def to_processed_document(self) -> ProcessedDocument:
        """Convert to a ``ProcessedDocument`` for unified memory storage."""
        return ProcessedDocument(
            doc_id=self.media_id,
            title=self.title,
            format=DocumentFormat.UNKNOWN,
            summary=self.summary,
            key_findings=self.key_facts,
            concepts=self.objects + self.diagram_notes,
            related_topics=self.topics,
            chunks=self.chunks,
            raw_text_length=len(self.transcript or self.ocr_text),
        )


# ===========================================================================
# Image Processor — fully functional (OCR + optional LLM description)
# ===========================================================================


_IMAGE_DESCRIBE_PROMPT = """\
You are Blix's media analyst. The OCR text extracted from an image is below.
This image may be a screenshot, diagram, chart, or photo.

Based on the OCR text, infer:
1. summary: 1-2 sentence description of what the image likely shows
2. objects: list of UI elements / components / labelled boxes mentioned
3. diagram_notes: list of architecture/flow notes if this looks like a
   diagram (connections like "A -> B", component names); empty list if N/A
4. topics: 2-5 short topic labels

Respond with ONLY a JSON object:
{{"summary": "...", "objects": ["..."], "diagram_notes": ["..."], "topics": ["..."]}}

OCR text:
{text}
"""


class ImageProcessor:
    """
    Processes images: OCR text extraction + optional LLM-based description.

    Parameters
    ----------
    llm:
        Optional LLM for interpreting OCR output (diagrams, screenshots).
        If ``None``, heuristic summary is used.
    ocr_lang:
        Tesseract language code.
    """

    def __init__(self, llm: Optional[LLMProvider] = None, ocr_lang: str = "eng") -> None:
        self._llm = llm
        self._ocr_lang = ocr_lang

    def process(self, path: Path) -> ProcessedMedia:
        """
        Run OCR + analysis on an image file.

        Raises
        ------
        RuntimeError
            If pytesseract/Pillow are unavailable or OCR fails entirely.
        """
        ocr_text = self._ocr(path)
        analysis = self._analyse(ocr_text, path.stem)

        media_id = _slugify(path.stem) + "_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        chunks = []
        if ocr_text.strip():
            chunks = [DocumentChunk(chunk_id="chunk_0", text=ocr_text.strip(), chunk_index=0)]

        result = ProcessedMedia(
            media_id=media_id,
            media_type=MediaType.IMAGE,
            title=path.stem,
            summary=analysis.get("summary", ""),
            ocr_text=ocr_text,
            objects=analysis.get("objects", []),
            diagram_notes=analysis.get("diagram_notes", []),
            topics=analysis.get("topics", []),
            chunks=chunks,
        )
        log.info("ImageProcessor: processed %s (%d OCR chars)", path.name, len(ocr_text))
        return result

    def _ocr(self, path: Path) -> str:
        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "ImageProcessor requires 'Pillow' and 'pytesseract' "
                "(plus the tesseract-ocr binary)."
            ) from exc

        try:
            with Image.open(path) as img:
                text = pytesseract.image_to_string(img, lang=self._ocr_lang)
            return text.strip()
        except Exception as exc:
            log.warning("ImageProcessor: OCR failed for %s (%s)", path.name, exc)
            return ""

    def _analyse(self, ocr_text: str, title: str) -> dict:
        if not ocr_text.strip():
            return {
                "summary": f"Image '{title}' contains no extractable text.",
                "objects": [], "diagram_notes": [], "topics": [],
            }
        if self._llm is not None:
            return self._llm_analyse(ocr_text, title)
        return self._heuristic_analyse(ocr_text, title)

    def _llm_analyse(self, ocr_text: str, title: str) -> dict:
        prompt = _IMAGE_DESCRIBE_PROMPT.format(text=ocr_text[:3000])
        try:
            raw = _strip_code_fence(self._llm.generate(prompt).strip())  # type: ignore[union-attr]
            data = json.loads(raw)
            return {
                "summary": str(data.get("summary", "")),
                "objects": list(data.get("objects", [])),
                "diagram_notes": list(data.get("diagram_notes", [])),
                "topics": list(data.get("topics", [])),
            }
        except Exception as exc:
            log.warning("ImageProcessor: LLM analysis failed (%s); using heuristic.", exc)
            return self._heuristic_analyse(ocr_text, title)

    def _heuristic_analyse(self, ocr_text: str, title: str) -> dict:
        lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        # Detect arrow-like diagram connections, e.g. "A -> B", "A→B"
        arrows = [l for l in lines if re.search(r"-+>|→|=>", l)]
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", ocr_text)
        freq: dict[str, int] = {}
        for w in words:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
        topics = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:5]]
        summary = f"Image '{title}' contains text" + (
            f" including: {', '.join(lines[:3])}" if lines else "."
        )
        return {
            "summary": summary[:300],
            "objects": lines[:10],
            "diagram_notes": arrows,
            "topics": topics,
        }


# ===========================================================================
# Audio Processor — pluggable transcription backend
# ===========================================================================


class TranscriptionBackend(ABC):
    """
    Extension point for speech-to-text.

    Implement this with `openai-whisper`, `faster-whisper`, a cloud ASR
    API, etc. and pass an instance to ``AudioProcessor``.
    """

    @abstractmethod
    def transcribe(self, path: Path) -> list[tuple[float, float, str]]:
        """
        Transcribe an audio file.

        Returns a list of (start_seconds, end_seconds, text) segments.
        """
        raise NotImplementedError


class NullTranscriptionBackend(TranscriptionBackend):
    """
    Offline placeholder backend. Returns an empty transcript and logs
    a warning explaining how to enable real transcription.

    This keeps ``AudioProcessor``/``VideoProcessor`` importable and
    testable without heavy ML dependencies, while making the missing
    capability explicit rather than silently wrong.
    """

    def transcribe(self, path: Path) -> list[tuple[float, float, str]]:
        log.warning(
            "NullTranscriptionBackend: no transcription performed for %s. "
            "Provide a real TranscriptionBackend (e.g. wrapping "
            "openai-whisper or faster-whisper) to AudioProcessor/"
            "VideoProcessor to enable audio understanding.",
            path.name,
        )
        return []


_AUDIO_ANALYSIS_PROMPT = """\
You are Blix's media analyst. Below is a transcript of an audio recording
(e.g. a lecture or meeting). Extract:

1. summary: 2-3 sentence summary
2. topics: 2-5 short topic labels (learning topics / subjects covered)
3. action_items: list of concrete action items or follow-ups mentioned
4. key_facts: list of 1-5 key factual statements from the recording

Respond with ONLY a JSON object:
{{"summary": "...", "topics": ["..."], "action_items": ["..."], "key_facts": ["..."]}}

Transcript:
{text}
"""


class AudioProcessor:
    """
    Processes audio files: transcription + topic/action-item extraction.

    Parameters
    ----------
    transcription_backend:
        Implementation of ``TranscriptionBackend``. Defaults to
        ``NullTranscriptionBackend`` (no-op, logs a warning).
    llm:
        Optional LLM for transcript analysis.
    """

    def __init__(
        self,
        transcription_backend: Optional[TranscriptionBackend] = None,
        llm: Optional[LLMProvider] = None,
    ) -> None:
        self._backend = transcription_backend or NullTranscriptionBackend()
        self._llm = llm

    def process(self, path: Path) -> ProcessedMedia:
        segments = self._backend.transcribe(path)
        transcript = " ".join(seg[2] for seg in segments).strip()
        analysis = self._analyse(transcript, path.stem)

        media_id = _slugify(path.stem) + "_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        chunks = self._chunk_transcript(transcript)

        result = ProcessedMedia(
            media_id=media_id,
            media_type=MediaType.AUDIO,
            title=path.stem,
            summary=analysis.get("summary", ""),
            transcript=transcript,
            topics=analysis.get("topics", []),
            action_items=analysis.get("action_items", []),
            key_facts=analysis.get("key_facts", []),
            segments=segments,
            chunks=chunks,
        )
        log.info(
            "AudioProcessor: processed %s (%d segments, %d transcript chars)",
            path.name, len(segments), len(transcript),
        )
        return result

    def _chunk_transcript(self, transcript: str, chunk_size: int = 1000) -> list[DocumentChunk]:
        if not transcript.strip():
            return []
        chunks = []
        for i in range(0, len(transcript), chunk_size):
            piece = transcript[i:i + chunk_size].strip()
            if piece:
                chunks.append(DocumentChunk(chunk_id=f"chunk_{len(chunks)}", text=piece, chunk_index=len(chunks)))
        return chunks

    def _analyse(self, transcript: str, title: str) -> dict:
        if not transcript.strip():
            return {
                "summary": f"No transcript available for '{title}'.",
                "topics": [], "action_items": [], "key_facts": [],
            }
        if self._llm is not None:
            return self._llm_analyse(transcript, title)
        return self._heuristic_analyse(transcript, title)

    def _llm_analyse(self, transcript: str, title: str) -> dict:
        prompt = _AUDIO_ANALYSIS_PROMPT.format(text=transcript[:6000])
        try:
            raw = _strip_code_fence(self._llm.generate(prompt).strip())  # type: ignore[union-attr]
            data = json.loads(raw)
            return {
                "summary": str(data.get("summary", "")),
                "topics": list(data.get("topics", [])),
                "action_items": list(data.get("action_items", [])),
                "key_facts": list(data.get("key_facts", [])),
            }
        except Exception as exc:
            log.warning("AudioProcessor: LLM analysis failed (%s); using heuristic.", exc)
            return self._heuristic_analyse(transcript, title)

    def _heuristic_analyse(self, transcript: str, title: str) -> dict:
        sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
        action_items = [s for s in sentences if re.search(r"\b(need to|should|will|todo|action item)\b", s, re.I)]
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", transcript)
        freq: dict[str, int] = {}
        for w in words:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
        topics = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:5]]
        summary = (sentences[0][:300] if sentences else f"Recording '{title}' processed.")
        return {
            "summary": summary,
            "topics": topics,
            "action_items": action_items[:5],
            "key_facts": sentences[:3],
        }


# ===========================================================================
# Video Processor — frame sampling + delegated audio/image processing
# ===========================================================================


class VideoProcessor:
    """
    Processes video files: samples frames (via ffmpeg if available),
    runs ``ImageProcessor`` on key frames, and delegates the audio track
    to ``AudioProcessor``.

    Parameters
    ----------
    image_processor:
        Used for frame analysis.
    audio_processor:
        Used for the audio track (requires ffmpeg to extract audio).
    frame_sample_seconds:
        Sample one frame every N seconds.
    max_frames:
        Cap on number of frames analysed (cost control).
    """

    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        frame_sample_seconds: float = 30.0,
        max_frames: int = 10,
    ) -> None:
        self._image_proc = image_processor or ImageProcessor()
        self._audio_proc = audio_processor or AudioProcessor()
        self._sample_seconds = frame_sample_seconds
        self._max_frames = max_frames

    def process(self, path: Path) -> ProcessedMedia:
        """
        Process a video file.

        Requires ``ffmpeg`` on PATH for frame/audio extraction. If
        ffmpeg is unavailable, returns a ``ProcessedMedia`` with an
        explanatory summary and no transcript/frames (graceful degradation).
        """
        if shutil.which("ffmpeg") is None:
            log.warning(
                "VideoProcessor: ffmpeg not found on PATH — video understanding "
                "disabled. Install ffmpeg to enable frame/audio extraction."
            )
            media_id = _slugify(path.stem) + "_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            return ProcessedMedia(
                media_id=media_id,
                media_type=MediaType.VIDEO,
                title=path.stem,
                summary=f"Video '{path.stem}' could not be processed: ffmpeg not available.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            frame_media = self._process_frames(path, tmpdir)
            audio_media = self._process_audio_track(path, tmpdir)

        media_id = _slugify(path.stem) + "_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        all_topics = list({*frame_media.get("topics", []), *audio_media.topics})
        chunks = audio_media.chunks + frame_media.get("chunks", [])

        summary_parts = []
        if audio_media.summary:
            summary_parts.append(audio_media.summary)
        if frame_media.get("summary"):
            summary_parts.append(frame_media["summary"])
        summary = " ".join(summary_parts) or f"Video '{path.stem}' processed."

        result = ProcessedMedia(
            media_id=media_id,
            media_type=MediaType.VIDEO,
            title=path.stem,
            summary=summary,
            transcript=audio_media.transcript,
            objects=frame_media.get("objects", []),
            topics=all_topics,
            action_items=audio_media.action_items,
            key_facts=audio_media.key_facts,
            segments=audio_media.segments,
            chunks=chunks,
        )
        log.info(
            "VideoProcessor: processed %s (%d frames, %d transcript chars)",
            path.name, frame_media.get("frame_count", 0), len(audio_media.transcript),
        )
        return result

    # ------------------------------------------------------------------
    # Frame extraction (ffmpeg)
    # ------------------------------------------------------------------

    def _process_frames(self, path: Path, tmpdir: Path) -> dict:
        pattern = tmpdir / "frame_%04d.jpg"
        cmd = [
            "ffmpeg", "-i", str(path),
            "-vf", f"fps=1/{self._sample_seconds}",
            "-vframes", str(self._max_frames),
            str(pattern), "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except Exception as exc:
            log.warning("VideoProcessor: frame extraction failed (%s)", exc)
            return {}

        frames = sorted(tmpdir.glob("frame_*.jpg"))
        objects: list[str] = []
        topics: set[str] = set()
        chunks: list[DocumentChunk] = []
        summaries: list[str] = []

        for i, frame_path in enumerate(frames):
            try:
                media = self._image_proc.process(frame_path)
            except RuntimeError:
                break  # OCR deps unavailable — stop trying further frames
            if media.summary:
                summaries.append(media.summary)
            objects.extend(media.objects)
            topics.update(media.topics)
            for c in media.chunks:
                chunks.append(DocumentChunk(
                    chunk_id=f"frame_{i}_{c.chunk_id}", text=c.text, chunk_index=len(chunks),
                ))

        return {
            "objects": objects[:20],
            "topics": list(topics),
            "chunks": chunks,
            "summary": " ".join(summaries[:3]),
            "frame_count": len(frames),
        }

    # ------------------------------------------------------------------
    # Audio track extraction (ffmpeg)
    # ------------------------------------------------------------------

    def _process_audio_track(self, path: Path, tmpdir: Path) -> ProcessedMedia:
        audio_path = tmpdir / "audio.wav"
        cmd = [
            "ffmpeg", "-i", str(path), "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000",
            str(audio_path), "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except Exception as exc:
            log.warning("VideoProcessor: audio extraction failed (%s)", exc)
            return ProcessedMedia(media_id="noop", media_type=MediaType.AUDIO, title=path.stem)

        return self._audio_proc.process(audio_path)


# ---------------------------------------------------------------------------
# Unified facade
# ---------------------------------------------------------------------------


class MediaProcessor:
    """
    Top-level facade dispatching to ``ImageProcessor`` / ``AudioProcessor`` /
    ``VideoProcessor`` based on file extension.
    """

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
    AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        transcription_backend: Optional[TranscriptionBackend] = None,
    ) -> None:
        self._image = ImageProcessor(llm=llm)
        self._audio = AudioProcessor(transcription_backend=transcription_backend, llm=llm)
        self._video = VideoProcessor(image_processor=self._image, audio_processor=self._audio)

    def process(self, path: Path) -> ProcessedMedia:
        ext = path.suffix.lower()
        if ext in self.IMAGE_EXTS:
            return self._image.process(path)
        if ext in self.AUDIO_EXTS:
            return self._audio.process(path)
        if ext in self.VIDEO_EXTS:
            return self._video.process(path)
        raise ValueError(f"Unsupported media format: {ext}")

    def can_process(self, path: Path) -> bool:
        ext = path.suffix.lower()
        return ext in (self.IMAGE_EXTS | self.AUDIO_EXTS | self.VIDEO_EXTS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", text.lower().replace(" ", "_"))


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
