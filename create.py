import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import logging
import hashlib
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    
    # Input directories
    train_clean_dir: str = "/gdata/fewahab/data/WSJ0-Full-raw/train"
    valid_clean_dir: str = "/gdata/fewahab/data/WSJ0-Full-raw/valid"
    test_clean_dir: str = "/gdata/fewahab/data/WSJ0-Full-raw/test"
    noise_dir: str = "/gdata/fewahab/data/WSJ0-Full-raw/NOISEX-92-16K"
    output_dir: str = "/gdata/fewahab/data/WSJO-full-wavdataset" 
    
    # Audio parameters
    sample_rate: int = 16000
    
    # MS-SNSD Standard: Normalize to -25 dBFS
    # -25 dBFS = 10^(-25/20) = 0.05623 RMS for normalized float audio
    target_level_dbfs: float = -25.0
    
    # SNR ranges
    train_snr_range: Tuple[float, float] = (-10.0, 20.0)
    valid_snr_range: Tuple[float, float] = (-10.0, 20.0)
    test_snr_levels: List[float] = field(default_factory=lambda: [-6.0, -3.0, 0.0, 3.0, 6.0])
    
    # Processing parameters
    random_seed: int = 42
    num_workers: int = 8
    batch_size: int = 100
    max_noise_repeat: int = 3
    max_retry_attempts: int = 50
    max_snr_error_db: float = 0.5  # Stricter tolerance
    
    # Clipping prevention
    max_amplitude: float = 0.95  # Leave headroom to prevent clipping
    
    @property
    def target_rms(self) -> float:
        """Convert dBFS to linear RMS value."""
        return 10 ** (self.target_level_dbfs / 20)


# =============================================================================
# MS-SNSD STANDARD MIXING FUNCTION
# =============================================================================

def mix_at_snr_mssnsd_standard(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    target_level_dbfs: float = -25.0,
    max_amplitude: float = 0.95,
    max_snr_error: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict]:
    """
    MS-SNSD Standard: Mix clean speech with noise at specified SNR.
    
    THIS IS THE GOLD STANDARD APPROACH used by:
    - Microsoft MS-SNSD dataset
    - NVIDIA Riva tutorials
    - Many state-of-the-art speech enhancement papers
    
    Algorithm:
    1. Normalize CLEAN to fixed level (-25 dBFS) FIRST
    2. Normalize NOISE to same fixed level (-25 dBFS)
    3. Scale noise to achieve target SNR
    4. Mix: noisy = clean_norm + noise_scaled
    5. If clipping occurs, scale ALL signals by same factor
    
    CRITICAL PROPERTY:
    - Clean speech ALWAYS has the same level regardless of SNR
    - This is essential for fair PESQ/STOI/MOS evaluation
    
    Args:
        clean: Clean speech signal (numpy array)
        noise: Noise signal (numpy array, will be trimmed/tiled to match clean)
        snr_db: Target Signal-to-Noise Ratio in dB
        target_level_dbfs: Target level in dBFS (default: -25 dBFS per MS-SNSD)
        max_amplitude: Maximum allowed amplitude (default: 0.95 to prevent clipping)
        max_snr_error: Maximum allowed SNR error in dB
        
    Returns:
        clean_final: Normalized clean signal
        noise_final: Scaled noise signal
        noisy_final: Mixed signal (clean_final + noise_final)
        actual_snr: Measured SNR after processing
        metadata: Dictionary with all processing parameters
    """
    
    # =========================================================================
    # Step 0: Match noise length to clean
    # =========================================================================
    if len(noise) < len(clean):
        num_repeats = int(np.ceil(len(clean) / len(noise)))
        if num_repeats > 3:
            raise ValueError(f"Noise too short: needs {num_repeats} repeats (max 3 allowed)")
        noise = np.tile(noise, num_repeats)
    noise = noise[:len(clean)]
    
    # =========================================================================
    # Step 1: Calculate target RMS from dBFS
    # =========================================================================
    target_rms = 10 ** (target_level_dbfs / 20)
    
    # =========================================================================
    # Step 2: Normalize CLEAN to target level (-25 dBFS)
    # =========================================================================
    clean_rms_original = np.sqrt(np.mean(clean ** 2))
    if clean_rms_original < 1e-10:
        raise ValueError(f"Clean signal is silent (RMS={clean_rms_original:.2e})")
    
    clean_scale = target_rms / clean_rms_original
    clean_norm = clean * clean_scale
    
    # Verify clean normalization
    clean_rms_norm = np.sqrt(np.mean(clean_norm ** 2))
    
    # =========================================================================
    # Step 3: Normalize NOISE to same target level (-25 dBFS)
    # =========================================================================
    noise_rms_original = np.sqrt(np.mean(noise ** 2))
    if noise_rms_original < 1e-10:
        raise ValueError(f"Noise signal is silent (RMS={noise_rms_original:.2e})")
    
    noise_scale = target_rms / noise_rms_original
    noise_norm = noise * noise_scale
    
    # =========================================================================
    # Step 4: Scale noise to achieve target SNR
    # =========================================================================
    # SNR = 20 * log10(clean_rms / noise_rms)
    # noise_rms_target = clean_rms / 10^(snr_db/20)
    # 
    # Since both clean_norm and noise_norm have RMS = target_rms:
    # noise_rms_target = target_rms / 10^(snr_db/20)
    # scale_factor = noise_rms_target / target_rms = 1 / 10^(snr_db/20)
    
    snr_scale = 1.0 / (10 ** (snr_db / 20))
    noise_scaled = noise_norm * snr_scale
    
    # =========================================================================
    # Step 5: Create mixture
    # =========================================================================
    noisy = clean_norm + noise_scaled
    
    # =========================================================================
    # Step 6: Verify SNR
    # =========================================================================
    noise_scaled_rms = np.sqrt(np.mean(noise_scaled ** 2))
    actual_snr = 20 * np.log10(clean_rms_norm / noise_scaled_rms)
    snr_error = abs(snr_db - actual_snr)
    
    if snr_error > max_snr_error:
        raise ValueError(f"SNR error too large: {snr_error:.3f} dB (target: {snr_db}, actual: {actual_snr:.3f})")
    
    # =========================================================================
    # Step 7: Handle clipping - scale ALL signals by SAME factor
    # =========================================================================
    max_val = np.max(np.abs(noisy))
    clipping_applied = False
    clipping_scale = 1.0
    
    if max_val > max_amplitude:
        clipping_scale = max_amplitude / max_val
        clean_norm = clean_norm * clipping_scale
        noise_scaled = noise_scaled * clipping_scale
        noisy = noisy * clipping_scale
        clipping_applied = True
        
        # Re-verify SNR after clipping (should be unchanged since same scale applied to all)
        clean_rms_final = np.sqrt(np.mean(clean_norm ** 2))
        noise_rms_final = np.sqrt(np.mean(noise_scaled ** 2))
        actual_snr = 20 * np.log10(clean_rms_final / noise_rms_final)
    
    # =========================================================================
    # Step 8: Final measurements
    # =========================================================================
    clean_rms_final = np.sqrt(np.mean(clean_norm ** 2))
    noise_rms_final = np.sqrt(np.mean(noise_scaled ** 2))
    noisy_rms_final = np.sqrt(np.mean(noisy ** 2))
    
    # Convert to dBFS for logging
    clean_level_dbfs = 20 * np.log10(clean_rms_final + 1e-10)
    noise_level_dbfs = 20 * np.log10(noise_rms_final + 1e-10)
    noisy_level_dbfs = 20 * np.log10(noisy_rms_final + 1e-10)
    
    metadata = {
        # Target parameters
        'target_level_dbfs': float(target_level_dbfs),
        'target_snr_db': float(snr_db),
        
        # Actual measurements
        'actual_snr_db': float(actual_snr),
        'snr_error_db': float(abs(snr_db - actual_snr)),
        
        # Signal levels (RMS)
        'clean_rms_original': float(clean_rms_original),
        'noise_rms_original': float(noise_rms_original),
        'clean_rms_final': float(clean_rms_final),
        'noise_rms_final': float(noise_rms_final),
        'noisy_rms_final': float(noisy_rms_final),
        
        # Signal levels (dBFS)
        'clean_level_dbfs': float(clean_level_dbfs),
        'noise_level_dbfs': float(noise_level_dbfs),
        'noisy_level_dbfs': float(noisy_level_dbfs),
        
        # Scale factors applied
        'clean_scale_factor': float(clean_scale * clipping_scale),
        'noise_scale_factor': float(noise_scale * snr_scale * clipping_scale),
        'snr_scale_factor': float(snr_scale),
        
        # Clipping info
        'clipping_applied': clipping_applied,
        'clipping_scale': float(clipping_scale),
        'max_amplitude': float(np.max(np.abs(noisy))),
    }
    
    return clean_norm, noise_scaled, noisy, actual_snr, metadata


# =============================================================================
# WORKER FUNCTION FOR PARALLEL PROCESSING
# =============================================================================

def _generate_pair_worker(
    clean_path: Path,
    noise_path: Path,
    snr_db: float,
    output_idx: int,
    split: str,
    seed: int,
    config: DatasetConfig
) -> Optional[Dict]:
    """
    Worker function for parallel processing of train/valid pairs.
    
    Uses MS-SNSD standard normalization:
    - Clean is normalized to -25 dBFS FIRST
    - Noise is scaled to achieve target SNR
    - Clean level is CONSTANT regardless of SNR
    """
    
    try:
        # Load audio files
        clean, sr_clean = sf.read(str(clean_path), dtype='float64')
        noise, sr_noise = sf.read(str(noise_path), dtype='float64')
        
        # Validate sample rates
        if sr_clean != config.sample_rate:
            raise ValueError(f"Clean sample rate mismatch: {sr_clean} != {config.sample_rate}")
        if sr_noise != config.sample_rate:
            raise ValueError(f"Noise sample rate mismatch: {sr_noise} != {config.sample_rate}")
        
        # Convert to mono if needed
        if len(clean.shape) > 1:
            clean = np.mean(clean, axis=1)
        if len(noise.shape) > 1:
            noise = np.mean(noise, axis=1)
        
        # Mix using MS-SNSD standard
        clean_norm, noise_scaled, noisy, actual_snr, mix_metadata = mix_at_snr_mssnsd_standard(
            clean=clean,
            noise=noise,
            snr_db=snr_db,
            target_level_dbfs=config.target_level_dbfs,
            max_amplitude=config.max_amplitude,
            max_snr_error=config.max_snr_error_db
        )
        
        # Create output directories
        output_clean_dir = Path(config.output_dir) / split / 'clean'
        output_noisy_dir = Path(config.output_dir) / split / 'noisy'
        output_clean_dir.mkdir(parents=True, exist_ok=True)
        output_noisy_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{split}_{output_idx:06d}.wav"
        
        # Save files as 16-bit PCM
        sf.write(
            str(output_clean_dir / filename),
            clean_norm.astype(np.float32),
            config.sample_rate,
            subtype='PCM_16'
        )
        
        sf.write(
            str(output_noisy_dir / filename),
            noisy.astype(np.float32),
            config.sample_rate,
            subtype='PCM_16'
        )
        
        # Build metadata
        metadata = {
            'index': output_idx,
            'filename': filename,
            'split': split,
            'clean_source': clean_path.name,
            'noise_source': noise_path.name,
            'duration_sec': float(len(clean_norm) / config.sample_rate),
            'num_samples': int(len(clean_norm)),
            **mix_metadata
        }
        
        return metadata
        
    except Exception as e:
        logging.error(f"Failed {split}_{output_idx:06d}: {e}")
        raise


# =============================================================================
# MAIN DATASET GENERATOR CLASS
# =============================================================================

class DatasetGenerator:
    """
    WSJ0 Speech Enhancement Dataset Generator
    
    Uses MS-SNSD Standard Normalization:
    - Clean speech is ALWAYS normalized to -25 dBFS
    - Clean level is CONSTANT across all SNR conditions
    - This is the SAFEST approach for PESQ, STOI, and subjective evaluation
    
    Output Structure:
        train/
            clean/  (all files normalized to -25 dBFS)
            noisy/
        valid/
            clean/  (all files normalized to -25 dBFS)
            noisy/
        test/
            snr_-06/
                clean/  (normalized to -25 dBFS, may be scaled if clipping)
                noisy/
            snr_-03/
                clean/
                noisy/
            snr_+00/
                clean/
                noisy/
            snr_+03/
                clean/
                noisy/
            snr_+06/
                clean/
                noisy/
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._setup_logging()
        np.random.seed(config.random_seed)
        
        self.logger.info("=" * 70)
        self.logger.info("WSJ0 Dataset Generator - MS-SNSD Standard Normalization")
        self.logger.info("=" * 70)
        self.logger.info(f"Target level: {config.target_level_dbfs} dBFS")
        self.logger.info(f"Target RMS: {config.target_rms:.6f}")
        self.logger.info("")
        
        # Collect files
        self.logger.info("Collecting files...")
        self.clean_files = {
            'train': self._collect_files(config.train_clean_dir),
            'valid': self._collect_files(config.valid_clean_dir),
            'test': self._collect_files(config.test_clean_dir)
        }
        self.noise_files = self._collect_files(config.noise_dir)
        
        for split, files in self.clean_files.items():
            self.logger.info(f"  {split}: {len(files)} files")
        self.logger.info(f"  noise: {len(self.noise_files)} files")
        
        # Validate files
        self._validate_files()
        
        # Compute file lengths
        self.logger.info("Computing file metadata...")
        self.clean_lengths = {
            'train': self._get_file_lengths(self.clean_files['train']),
            'valid': self._get_file_lengths(self.clean_files['valid']),
            'test': self._get_file_lengths(self.clean_files['test'])
        }
        self.noise_lengths = self._get_file_lengths(self.noise_files)
        self.logger.info("Done\n")
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('DatasetGenerator')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(log_dir / 'generation.log', mode='w')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _collect_files(self, directory: str) -> List[Path]:
        """Collect and validate WAV files from directory."""
        directory = Path(directory)
        if not directory.exists():
            self.logger.error(f"Directory not found: {directory}")
            return []
        
        files = sorted(list(directory.rglob('*.wav')))
        valid_files = []
        
        for f in tqdm(files, desc=f"Validating {directory.name}", leave=False):
            try:
                info = sf.info(str(f))
                if info.frames > 0 and info.samplerate > 0:
                    duration = info.frames / info.samplerate
                    if duration >= 0.1:  # At least 100ms
                        valid_files.append(f)
            except Exception:
                pass
        
        return valid_files
    
    def _get_file_lengths(self, files: List[Path]) -> Dict[Path, int]:
        """Get lengths of all audio files."""
        lengths = {}
        
        for f in tqdm(files, desc="Computing lengths", leave=False):
            try:
                info = sf.info(str(f))
                lengths[f] = info.frames
            except Exception as e:
                raise ValueError(f"Cannot read {f.name}: {e}")
        
        return lengths
    
    def _validate_files(self):
        """Validate all input files."""
        errors = []
        
        # Check for empty directories
        for split, files in self.clean_files.items():
            if len(files) == 0:
                errors.append(f"No {split} files found")
        if len(self.noise_files) == 0:
            errors.append("No noise files found")
        
        if errors:
            for e in errors:
                self.logger.error(e)
            raise ValueError("Insufficient input files")
        
        # Check sample rates
        self.logger.info(f"Verifying sample rate ({self.config.sample_rate} Hz)...")
        wrong_sr = []
        
        all_files = (
            self.clean_files['train'] +
            self.clean_files['valid'] +
            self.clean_files['test'] +
            self.noise_files
        )
        
        for f in tqdm(all_files, desc="Checking sample rates", leave=False):
            try:
                info = sf.info(str(f))
                if info.samplerate != self.config.sample_rate:
                    wrong_sr.append(f"{f.name}: {info.samplerate} Hz")
            except Exception:
                wrong_sr.append(f"{f.name}: ERROR")
        
        if wrong_sr:
            self.logger.error(f"Files with wrong sample rate: {wrong_sr[:5]}...")
            raise ValueError(f"{len(wrong_sr)} files not at {self.config.sample_rate} Hz")
        
        self.logger.info("All files validated successfully")
    
    def _validate_noise_length(self, noise_path: Path, clean_length: int) -> bool:
        """Check if noise is long enough for the clean signal."""
        noise_length = self.noise_lengths.get(noise_path, 0)
        if noise_length == 0:
            return False
        max_length = noise_length * self.config.max_noise_repeat
        return max_length >= int(clean_length * 0.95)
    
    def _find_suitable_noise(self, clean_path: Path, rng: np.random.RandomState, split: str) -> Path:
        """Find a suitable noise file for the given clean file."""
        clean_length = self.clean_lengths[split].get(clean_path)
        if clean_length is None:
            raise RuntimeError(f"Clean file {clean_path.name} not in cache")
        
        for _ in range(self.config.max_retry_attempts):
            noise_idx = rng.randint(0, len(self.noise_files))
            noise_path = self.noise_files[noise_idx]
            if self._validate_noise_length(noise_path, clean_length):
                return noise_path
        
        raise RuntimeError(f"No suitable noise found for {clean_path.name}")
    
    def generate_train_set(self):
        """Generate training set."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("GENERATING TRAINING SET")
        self.logger.info("=" * 70)
        self.logger.info(f"SNR range: {self.config.train_snr_range}")
        self.logger.info(f"Clean level: {self.config.target_level_dbfs} dBFS (constant)")
        
        clean_files = self.clean_files['train']
        tasks = []
        
        for idx, clean_path in enumerate(clean_files):
            sample_seed = self.config.random_seed + idx
            rng = np.random.RandomState(sample_seed)
            snr_db = rng.uniform(*self.config.train_snr_range)
            noise_path = self._find_suitable_noise(clean_path, rng, 'train')
            tasks.append((clean_path, noise_path, snr_db, idx, 'train', sample_seed))
        
        metadata_list = self._process_tasks(tasks, 'train')
        
        if len(metadata_list) < len(clean_files):
            raise RuntimeError(f"Training set incomplete: {len(metadata_list)}/{len(clean_files)}")
        
        self._save_metadata(metadata_list, 'train')
        self.logger.info(f"Training set complete: {len(metadata_list)} files\n")
    
    def generate_valid_set(self):
        """Generate validation set."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("GENERATING VALIDATION SET")
        self.logger.info("=" * 70)
        self.logger.info(f"SNR range: {self.config.valid_snr_range}")
        self.logger.info(f"Clean level: {self.config.target_level_dbfs} dBFS (constant)")
        
        clean_files = self.clean_files['valid']
        tasks = []
        
        for idx, clean_path in enumerate(clean_files):
            sample_seed = self.config.random_seed + 100000 + idx
            rng = np.random.RandomState(sample_seed)
            snr_db = rng.uniform(*self.config.valid_snr_range)
            noise_path = self._find_suitable_noise(clean_path, rng, 'valid')
            tasks.append((clean_path, noise_path, snr_db, idx, 'valid', sample_seed))
        
        metadata_list = self._process_tasks(tasks, 'valid')
        
        if len(metadata_list) < len(clean_files):
            raise RuntimeError(f"Validation set incomplete: {len(metadata_list)}/{len(clean_files)}")
        
        self._save_metadata(metadata_list, 'valid')
        self.logger.info(f"Validation set complete: {len(metadata_list)} files\n")
    
    def generate_test_set(self):
        """
        Generate test set with separate folders for each SNR level.
        
        MS-SNSD Standard:
        - Clean is normalized to -25 dBFS FIRST
        - Noise is scaled to achieve target SNR
        - Clean level is CONSTANT (unless clipping prevention is needed)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("GENERATING TEST SET")
        self.logger.info("=" * 70)
        self.logger.info(f"SNR levels: {self.config.test_snr_levels}")
        self.logger.info(f"Clean level: {self.config.target_level_dbfs} dBFS (constant)")
        self.logger.info("")
        self.logger.info("IMPORTANT: Clean files have CONSTANT level across all SNRs")
        self.logger.info("This is essential for fair PESQ/STOI/MOS evaluation")
        
        clean_files = self.clean_files['test']
        test_dir = Path(self.config.output_dir) / 'test'
        all_metadata = []
        
        for snr_db in self.config.test_snr_levels:
            self.logger.info(f"\nProcessing SNR {snr_db:+.1f} dB...")
            
            # Create directories
            snr_folder = test_dir / f'snr_{int(snr_db):+03d}'
            clean_dir = snr_folder / 'clean'
            noisy_dir = snr_folder / 'noisy'
            clean_dir.mkdir(parents=True, exist_ok=True)
            noisy_dir.mkdir(parents=True, exist_ok=True)
            
            snr_metadata = []
            
            for idx in tqdm(range(len(clean_files)), desc=f"SNR {snr_db:+.1f} dB"):
                success = False
                last_error = None
                
                for attempt in range(self.config.max_retry_attempts):
                    try:
                        # Deterministic noise selection
                        hash_input = f"{idx}_{int(snr_db)}_{attempt}"
                        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                        noise_idx = hash_val % len(self.noise_files)
                        noise_path = self.noise_files[noise_idx]
                        
                        clean_path = clean_files[idx]
                        clean_length = self.clean_lengths['test'][clean_path]
                        
                        # Check noise length
                        if not self._validate_noise_length(noise_path, clean_length):
                            continue
                        
                        # Load files
                        clean, sr = sf.read(str(clean_path), dtype='float64')
                        noise, sr_noise = sf.read(str(noise_path), dtype='float64')
                        
                        # Convert to mono
                        if len(clean.shape) > 1:
                            clean = np.mean(clean, axis=1)
                        if len(noise.shape) > 1:
                            noise = np.mean(noise, axis=1)
                        
                        # Validate sample rates
                        if sr != self.config.sample_rate or sr_noise != self.config.sample_rate:
                            raise ValueError(f"Sample rate mismatch")
                        
                        # Mix using MS-SNSD standard
                        clean_norm, noise_scaled, noisy, actual_snr, mix_metadata = mix_at_snr_mssnsd_standard(
                            clean=clean,
                            noise=noise,
                            snr_db=snr_db,
                            target_level_dbfs=self.config.target_level_dbfs,
                            max_amplitude=self.config.max_amplitude,
                            max_snr_error=self.config.max_snr_error_db
                        )
                        
                        # Save files
                        filename = f"test_{idx:04d}.wav"
                        
                        sf.write(
                            str(clean_dir / filename),
                            clean_norm.astype(np.float32),
                            self.config.sample_rate,
                            subtype='PCM_16'
                        )
                        
                        sf.write(
                            str(noisy_dir / filename),
                            noisy.astype(np.float32),
                            self.config.sample_rate,
                            subtype='PCM_16'
                        )
                        
                        # Build metadata
                        snr_metadata.append({
                            'index': idx,
                            'filename': filename,
                            'snr_level': int(snr_db),
                            'clean_source': clean_path.name,
                            'noise_source': noise_path.name,
                            'duration_sec': float(len(clean_norm) / self.config.sample_rate),
                            'num_samples': int(len(clean_norm)),
                            **mix_metadata
                        })
                        
                        success = True
                        break
                        
                    except Exception as e:
                        last_error = e
                        continue
                
                if not success:
                    raise RuntimeError(
                        f"Failed test_{idx:04d} at SNR {snr_db:+.1f} dB: {last_error}"
                    )
            
            # Verify completeness
            if len(snr_metadata) < len(clean_files):
                raise RuntimeError(
                    f"Test SNR {snr_db:+.1f} dB incomplete: {len(snr_metadata)}/{len(clean_files)}"
                )
            
            # Save metadata for this SNR level
            self._save_test_snr_metadata(snr_metadata, snr_folder)
            all_metadata.extend(snr_metadata)
            
            # Log statistics
            clean_levels = [m['clean_level_dbfs'] for m in snr_metadata]
            self.logger.info(f"  Clean level: {np.mean(clean_levels):.2f} {np.std(clean_levels):.2f} dBFS")
            self.logger.info(f"  Files saved: {len(snr_metadata)}")
        
        # Save combined metadata
        self._save_combined_test_metadata(all_metadata)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("TEST SET COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Total: {len(all_metadata)} pairs across {len(self.config.test_snr_levels)} SNR levels")
    
    def _process_tasks(self, tasks: List[Tuple], split: str) -> List[Dict]:
        """Process tasks in parallel."""
        metadata_list = []
        num_batches = int(np.ceil(len(tasks) / self.config.batch_size))
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            for batch_idx in range(num_batches):
                start = batch_idx * self.config.batch_size
                end = min((batch_idx + 1) * self.config.batch_size, len(tasks))
                batch = tasks[start:end]
                
                futures = [
                    executor.submit(_generate_pair_worker, *task, self.config)
                    for task in batch
                ]
                
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Batch {batch_idx + 1}/{num_batches}",
                    leave=False
                ):
                    result = future.result()
                    if result:
                        metadata_list.append(result)
        
        return metadata_list
    
    def _save_metadata(self, metadata_list: List[Dict], split: str):
        """Save metadata for train/valid splits."""
        output_dir = Path(self.config.output_dir) / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not metadata_list:
            return
        
        metadata_list = sorted(metadata_list, key=lambda x: x['index'])
        
        # Save CSV
        df = pd.DataFrame(metadata_list)
        df.to_csv(str(output_dir / 'metadata.csv'), index=False)
        
        # Save JSON
        with open(str(output_dir / 'metadata.json'), 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        # Save statistics
        snr_values = [m['actual_snr_db'] for m in metadata_list]
        clean_levels = [m['clean_level_dbfs'] for m in metadata_list]
        
        stats = {
            'split': split,
            'num_samples': len(metadata_list),
            'normalization': {
                'method': 'MS-SNSD Standard',
                'target_level_dbfs': self.config.target_level_dbfs,
                'description': 'Clean normalized to fixed level FIRST, then noise scaled for SNR'
            },
            'snr_statistics': {
                'min': float(np.min(snr_values)),
                'max': float(np.max(snr_values)),
                'mean': float(np.mean(snr_values)),
                'std': float(np.std(snr_values))
            },
            'clean_level_statistics': {
                'mean_dbfs': float(np.mean(clean_levels)),
                'std_dbfs': float(np.std(clean_levels)),
                'description': 'Should be close to target with small std'
            }
        }
        
        with open(str(output_dir / 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _save_test_snr_metadata(self, metadata_list: List[Dict], snr_folder: Path):
        """Save metadata for a single test SNR level."""
        if not metadata_list:
            return
        
        metadata_list = sorted(metadata_list, key=lambda x: x['index'])
        
        # Save CSV and JSON
        df = pd.DataFrame(metadata_list)
        df.to_csv(str(snr_folder / 'metadata.csv'), index=False)
        
        with open(str(snr_folder / 'metadata.json'), 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        # Save statistics
        snr_values = [m['actual_snr_db'] for m in metadata_list]
        clean_levels = [m['clean_level_dbfs'] for m in metadata_list]
        
        stats = {
            'snr_level': metadata_list[0]['snr_level'],
            'num_samples': len(metadata_list),
            'normalization': {
                'method': 'MS-SNSD Standard',
                'target_level_dbfs': self.config.target_level_dbfs
            },
            'actual_snr_statistics': {
                'target': float(metadata_list[0]['target_snr_db']),
                'mean': float(np.mean(snr_values)),
                'std': float(np.std(snr_values)),
                'min': float(np.min(snr_values)),
                'max': float(np.max(snr_values))
            },
            'clean_level_statistics': {
                'mean_dbfs': float(np.mean(clean_levels)),
                'std_dbfs': float(np.std(clean_levels))
            }
        }
        
        with open(str(snr_folder / 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _save_combined_test_metadata(self, all_metadata: List[Dict]):
        """Save combined metadata for all test SNR levels."""
        output_dir = Path(self.config.output_dir) / 'test'
        
        if not all_metadata:
            return
        
        all_metadata = sorted(all_metadata, key=lambda x: (x['snr_level'], x['index']))
        
        # Save CSV and JSON
        df = pd.DataFrame(all_metadata)
        df.to_csv(str(output_dir / 'metadata_all.csv'), index=False)
        
        with open(str(output_dir / 'metadata_all.json'), 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Overall statistics
        stats = {
            'split': 'test',
            'total_samples': len(all_metadata),
            'snr_levels': self.config.test_snr_levels,
            'samples_per_snr': len(all_metadata) // len(self.config.test_snr_levels),
            'normalization': {
                'method': 'MS-SNSD Standard',
                'target_level_dbfs': self.config.target_level_dbfs,
                'description': 'Clean normalized to fixed level, constant across all SNRs'
            },
            'snr_statistics': {}
        }
        
        for snr_db in self.config.test_snr_levels:
            snr_samples = [m for m in all_metadata if m['snr_level'] == int(snr_db)]
            snr_values = [m['actual_snr_db'] for m in snr_samples]
            clean_levels = [m['clean_level_dbfs'] for m in snr_samples]
            
            stats['snr_statistics'][f'snr_{int(snr_db):+03d}'] = {
                'target_snr': float(snr_db),
                'actual_snr_mean': float(np.mean(snr_values)),
                'actual_snr_std': float(np.std(snr_values)),
                'clean_level_mean_dbfs': float(np.mean(clean_levels)),
                'clean_level_std_dbfs': float(np.std(clean_levels)),
                'num_samples': len(snr_samples)
            }
        
        with open(str(output_dir / 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point."""
    
    config = DatasetConfig()
    
    print("\n" + "=" * 70)
    print("WSJ0 SPEECH ENHANCEMENT DATASET GENERATOR")
    print("MS-SNSD Standard Normalization")
    print("=" * 70)
    print(f"\nOutput Directory: {config.output_dir}")
    print(f"Sample Rate: {config.sample_rate} Hz")
    print(f"\n--- NORMALIZATION (MS-SNSD STANDARD) ---")
    print(f"Target Level: {config.target_level_dbfs} dBFS")
    print(f"Target RMS: {config.target_rms:.6f}")
    print(f"Method: Normalize CLEAN to fixed level FIRST, then scale noise for SNR")
    print(f"\n--- SNR CONFIGURATION ---")
    print(f"Train SNR Range: {config.train_snr_range} dB")
    print(f"Valid SNR Range: {config.valid_snr_range} dB")
    print(f"Test SNR Levels: {config.test_snr_levels} dB")
    print(f"\n--- PROCESSING ---")
    print(f"Workers: {config.num_workers}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Max Amplitude: {config.max_amplitude}")
    print("\n" + "=" * 70)
    print("\nKEY PROPERTY: Clean level is CONSTANT regardless of SNR")
    print("This ensures fair PESQ, STOI, and MOS evaluation")
    print("\n" + "=" * 70 + "\n")
    
    try:
        generator = DatasetGenerator(config)
        generator.generate_train_set()
        generator.generate_valid_set()
        generator.generate_test_set()
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"\nDataset saved to: {config.output_dir}")
        print(f"\nStructure:")
        print(f"  train/")
        print(f"    clean/  (normalized to {config.target_level_dbfs} dBFS)")
        print(f"    noisy/")
        print(f"  valid/")
        print(f"    clean/  (normalized to {config.target_level_dbfs} dBFS)")
        print(f"    noisy/")
        print(f"  test/")
        for snr in config.test_snr_levels:
            print(f"    snr_{int(snr):+03d}/")
            print(f"      clean/  (normalized to {config.target_level_dbfs} dBFS)")
            print(f"      noisy/")
        print("\n" + "=" * 70)
        print("\nNORMALIZATION VERIFICATION:")
        print("- All clean files should have level   -25 dBFS")
        print("- Clean level should be CONSTANT across all SNR levels")
        print("- Check statistics.json for verification")
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n{'=' * 70}")
        print("GENERATION FAILED")
        print(f"{'=' * 70}")
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())