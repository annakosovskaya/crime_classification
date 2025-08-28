"""
Video processing module for frame extraction and sliding window processing.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator, NamedTuple
import logging

logger = logging.getLogger(__name__)


class VideoInfo(NamedTuple):
    """Container for video information."""
    video_path: str
    category: str
    total_frames: int
    fps: float
    duration_seconds: float
    width: int
    height: int


class WindowInfo(NamedTuple):
    """Container for sliding window information."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    window_id: str


class VideoProcessor:
    """Handles video frame extraction and sliding window processing."""
    
    def __init__(self, temp_dir: str = "tmp"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    def get_video_info(self, video_path: str, category: str = "unknown") -> Optional[VideoInfo]:
        """Extract basic information from video file."""
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            return VideoInfo(
                video_path=video_path,
                category=category,
                total_frames=total_frames,
                fps=fps,
                duration_seconds=duration_seconds,
                width=width,
                height=height
            )
        finally:
            cap.release()
    
    def generate_sliding_windows(
        self, 
        video_info: VideoInfo, 
        window_size_seconds: float,
        stride_seconds: float,
        min_window_size_seconds: float = 1.0 # Default to 1s
    ) -> List[WindowInfo]:
        """
        Generate sliding window specifications for a video.
        Ensures the final window is full-length by shifting its start time.
        """
        windows = []
        
        if video_info.duration_seconds < min_window_size_seconds:
            logger.warning(f"Video {video_info.video_path} is too short ({video_info.duration_seconds:.2f}s)")
            
        if video_info.duration_seconds < window_size_seconds:
            logger.warning(
                f"Video {video_info.video_path} ({video_info.duration_seconds:.2f}s) "
                f"is shorter than the window size ({window_size_seconds:.2f}s). "
                "Processing one window for the entire video."
            )
            # Add a single window covering the whole video
            end_time = video_info.duration_seconds
            start_frame = 0
            end_frame = video_info.total_frames
            window_id = f"{Path(video_info.video_path).stem}_window_000"
            windows.append(WindowInfo(
                start_frame=start_frame, end_frame=end_frame,
                start_time=0.0, end_time=end_time,
                duration=end_time, window_id=window_id
            ))
            return windows

        current_time = 0.0
        window_index = 0
        
        # Generate windows with a fixed stride
        while current_time + window_size_seconds <= video_info.duration_seconds:
            end_time = current_time + window_size_seconds
            start_frame = int(current_time * video_info.fps)
            end_frame = int(end_time * video_info.fps)
            window_id = f"{Path(video_info.video_path).stem}_window_{window_index:03d}"
            
            windows.append(WindowInfo(
                start_frame=start_frame, end_frame=end_frame,
                start_time=current_time, end_time=end_time,
                duration=window_size_seconds, window_id=window_id
            ))
            
            window_index += 1
            current_time += stride_seconds

        # Add the last window, ensuring it is full-length and flush with the end
        last_window_start_time = video_info.duration_seconds - window_size_seconds
        if not windows or windows[-1].start_time < last_window_start_time - 1e-9: # Check if last window is not already there
             end_time = video_info.duration_seconds
             start_time = last_window_start_time
             start_frame = int(start_time * video_info.fps)
             end_frame = video_info.total_frames
             window_id = f"{Path(video_info.video_path).stem}_window_{window_index:03d}"

             windows.append(WindowInfo(
                 start_frame=start_frame, end_frame=end_frame,
                 start_time=start_time, end_time=end_time,
                 duration=window_size_seconds, window_id=window_id
             ))

        return windows
    
    def extract_frames_from_window(
        self,
        video_info: VideoInfo,
        window: WindowInfo,
        model_frames: int
    ) -> List[np.ndarray]:
        """
        Extract exactly `model_frames` evenly spaced frames from `window` using
        a single VideoCapture with sequential grab()/read() where possible.
        This minimizes random seeks (cap.set) to improve performance.

        Possible improvement: use a single VideoCapture for all windows in a video.
        """
        cap = cv2.VideoCapture(video_info.video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_info.video_path}")
            return []

        frames: List[np.ndarray] = []
        try:
            # Target indices in [start_frame, end_frame-1], inclusive ends via endpoint=True
            idx_f = np.linspace(
                window.start_frame,
                window.end_frame - 1,
                num=model_frames,
                endpoint=True,
                dtype=float,
            )
            frame_indices = np.rint(idx_f).astype(int)
            frame_indices = np.clip(frame_indices, window.start_frame, window.end_frame - 1)

            # Single seek to the first target
            first_idx = int(frame_indices[0])
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
            current_next = first_idx  # frame index expected on the next read()

            prev_idx: Optional[int] = None
            for target in frame_indices:
                # If the rounded index repeats, duplicate last extracted frame
                if prev_idx is not None and target == prev_idx:
                    frames.append(frames[-1])
                    continue

                # If we need to go backwards (rare), do an explicit seek
                if target < current_next:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
                    current_next = int(target)

                # Advance with grab() so that the next read() returns the desired frame
                skips_needed = int(target - current_next)
                reached_eof = False
                for _ in range(max(0, skips_needed)):
                    if not cap.grab():  # EOF while advancing
                        reached_eof = True
                        break

                if reached_eof:
                    # Could not reach target; pad or abort if nothing read yet
                    if frames:
                        frames.append(frames[-1])
                        prev_idx = int(target)
                        current_next = int(target) + 1
                        continue
                    else:
                        logger.warning(f"Could not read any frames for window {window.window_id}")
                        break

                # Read the target frame
                success, frame = cap.read()
                if success and frame is not None:
                    frames.append(frame)
                    prev_idx = int(target)
                    current_next = int(target) + 1
                else:
                    # Pad with last successful frame if available
                    if frames:
                        frames.append(frames[-1])
                        prev_idx = int(target)
                        current_next = int(target) + 1
                    else:
                        logger.warning(f"Could not read any frames for window {window.window_id}")
                        break

            # Ensure exact length (pad with the last available frame if short)
            if len(frames) < model_frames:
                if frames:
                    last = frames[-1]
                    frames.extend([last] * (model_frames - len(frames)))
                else:
                    # No frames at all — return empty (already logged)
                    return []

            logger.debug(
                f"Extracted {len(frames)} frames for window {window.window_id} via sequential grab/read."
            )
        except Exception as e:
            logger.error(f"Error extracting frames from window {window.window_id}: {e}")
        finally:
            cap.release()

        return frames


class SlidingWindowProcessor:
    """Orchestrates sliding window processing for videos."""
    
    def __init__(self, video_processor: VideoProcessor):
        self.video_processor = video_processor
    
    def process_video_windows(
        self,
        video_info: VideoInfo,
        window_size_seconds: float,
        stride_seconds: float,
        min_window_size_seconds: float,
        model_frames: int
    ) -> Iterator[Tuple[WindowInfo, List[np.ndarray]]]:
        """Process video using sliding windows and yield frames extracted efficiently."""
        windows = self.video_processor.generate_sliding_windows(
            video_info, window_size_seconds, stride_seconds, min_window_size_seconds
        )

        logger.info(f"Generated {len(windows)} windows for video {video_info.video_path}")

        for window in windows:
            frames = self.video_processor.extract_frames_from_window(
                video_info, window, model_frames=model_frames
            )

            if frames:
                yield window, frames
            else:
                logger.warning(f"No frames extracted for window {window.window_id}.")

