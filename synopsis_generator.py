"""
synopsis_generator.py — Severity-Ordered Evidence Reel Synthesis

Takes reranked FCVAR results and stitches a severity-ordered
'evidence reel' MP4 using ffmpeg-python.

Segment selection logic
-----------------------
1. Filter  : drop results where R_final < min_r_final (default 0.5)
2. Window  : extract clip_duration/2 seconds before and after each
             peak fusion frame (temporal_window centred on timestamp)
3. Sort    : order segments by F_physical descending — severity first,
             NOT chronological order
4. Stitch  : concatenate via ffmpeg concat demuxer → severity_synopsis.mp4

Track-ID Highlighting
---------------------
If highlight_track_id is given, the function reports which segments
contain sightings of that specific BotSORT track ID so the operator
can cross-reference with the annotated output video.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import ffmpeg


def generate_evidence_reel(
    results:            List[Dict[str, Any]],
    output_path:        str            = "severity_synopsis.mp4",
    clip_duration:      float          = 5.0,
    min_r_final:        float          = 0.5,
    highlight_track_id: Optional[int]  = None,
) -> str:
    """
    Synthesise a severity-ordered evidence reel from FCVAR search results.

    Parameters
    ----------
    results :
        Reranked result list from FCVARManager.search().  Each item must
        have keys: video_path, timestamp_sec, f_physical, r_final, track_ids.
    output_path :
        Destination file for the stitched MP4.
    clip_duration :
        Total seconds to extract around each peak frame (half before, half after).
    min_r_final :
        Minimum R_final score to include a segment (default 0.5).
    highlight_track_id :
        Optional BotSORT track ID.  If provided, the function prints which
        segments contain that person, enabling easy cross-reference with the
        annotated output video.

    Returns
    -------
    Absolute path to the generated synopsis MP4.

    Raises
    ------
    ValueError  : if no results pass the R_final threshold.
    RuntimeError: if ffmpeg fails during extraction or concatenation.
    """
    # ---------------------------------------------------------------
    # 1. Filter
    # ---------------------------------------------------------------
    filtered = [r for r in results if r["r_final"] >= min_r_final]
    if not filtered:
        raise ValueError(
            f"No results passed R_final threshold ({min_r_final}). "
            "Lower the threshold or re-run the search."
        )

    # ---------------------------------------------------------------
    # 2. Temporal windowing
    # ---------------------------------------------------------------
    half = clip_duration / 2.0
    segments: List[Dict[str, Any]] = []

    for r in filtered:
        t_peak  = float(r["timestamp_sec"])
        t_start = max(0.0, round(t_peak - half, 3))
        t_end   = round(t_peak + half, 3)
        segments.append({
            "video_path": r["video_path"],
            "t_start"   : t_start,
            "t_end"     : t_end,
            "duration"  : round(t_end - t_start, 3),
            "f_physical": r["f_physical"],
            "r_final"   : r["r_final"],
            "track_ids" : r.get("track_ids", []),
            "timestamp" : t_peak,
        })

    # ---------------------------------------------------------------
    # 3. Severity sort (F_physical descending — NOT chronological)
    # ---------------------------------------------------------------
    segments.sort(key=lambda s: s["f_physical"], reverse=True)

    # ---------------------------------------------------------------
    # 4. Extract individual clips via ffmpeg
    # ---------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="hyvis_synopsis_")
    clip_paths: List[str] = []

    print(f"Extracting {len(segments)} clip segment(s)…")
    for i, seg in enumerate(segments):
        clip_path = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")
        try:
            (
                ffmpeg
                .input(seg["video_path"], ss=seg["t_start"], t=seg["duration"])
                .output(
                    clip_path,
                    vcodec="libx264",
                    acodec="aac",
                    loglevel="error",
                )
                .run(overwrite_output=True, quiet=True)
            )
            clip_paths.append(clip_path)
            print(
                f"  Segment {i+1:02d} | t={seg['t_start']:.1f}s–{seg['t_end']:.1f}s "
                f"| F_phys={seg['f_physical']:.3f} | R={seg['r_final']:.3f}"
            )
        except ffmpeg.Error as exc:
            print(f"  Warning: ffmpeg failed for segment {i} — {exc.stderr.decode()}")

    if not clip_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("All ffmpeg segment extractions failed.")

    # ---------------------------------------------------------------
    # 5. Write concat list and stitch
    # ---------------------------------------------------------------
    concat_file = os.path.join(tmp_dir, "concat_list.txt")
    with open(concat_file, "w") as f:
        for cp in clip_paths:
            # ffmpeg concat demuxer requires absolute paths or same-dir relative
            f.write(f"file '{os.path.abspath(cp)}'\n")

    try:
        (
            ffmpeg
            .input(concat_file, format="concat", safe=0)
            .output(output_path, c="copy", loglevel="error")
            .run(overwrite_output=True, quiet=True)
        )
    except ffmpeg.Error as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            f"ffmpeg concatenation failed: {exc.stderr.decode()}"
        )

    # ---------------------------------------------------------------
    # 6. Cleanup temp directory
    # ---------------------------------------------------------------
    shutil.rmtree(tmp_dir, ignore_errors=True)

    abs_output = os.path.abspath(output_path)
    print(f"\n✓ Evidence reel saved: '{abs_output}'")
    print(f"  Segments: {len(clip_paths)} | Sorted by F_physical (severity-first)")

    # ---------------------------------------------------------------
    # 7. Track-ID highlight report
    # ---------------------------------------------------------------
    if highlight_track_id is not None:
        matched = [
            (i + 1, s)
            for i, s in enumerate(segments)
            if highlight_track_id in s["track_ids"]
        ]
        if matched:
            print(
                f"\n  Track ID {highlight_track_id} appears in "
                f"{len(matched)} segment(s):"
            )
            for seg_num, s in matched:
                print(
                    f"    Segment {seg_num:02d} | t={s['t_start']:.1f}–"
                    f"{s['t_end']:.1f}s | F_phys={s['f_physical']:.3f}"
                )
        else:
            print(
                f"\n  Track ID {highlight_track_id} was NOT found "
                "in any selected segment."
            )

    return abs_output
