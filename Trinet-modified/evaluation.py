#!/usr/bin/env python
# coding: utf-8

import numpy as np
from module import *
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import librosa
from tqdm import tqdm
import soundfile as sf
from datetime import datetime

# ============================================
# CONFIGURATION - HARDCODED FOR SERVER
# ============================================
class Config:
    # Model selection - Choose one of: 'best', 'best_pesq', 'last'
    # 'best'      : checkpoint_best.pt (best validation loss)
    # 'best_pesq' : checkpoint_best_pesq.pt (best PESQ score)
    # 'last'      : checkpoint_last.pt (most recent epoch)
    model_type = 'last'

    # Server paths - MODIFY THESE FOR YOUR SERVER
    checkpoint_dir = '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/B3/saved_model'
    test_data_dir = '/gdata/fewahab/data/VoicebanK-demand-16K/test'
    enhanced_audio_dir = '/gdata/fewahab/My_5th_pap/Ab4-BSRNN/B3/enhanced_audio'

    # Evaluation settings
    save_enhanced_audio = True  # Save enhanced audio files
    n_fft = 512
    hop_length = 128

args = Config()


@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=512, hop=128, save_tracks=False):
    name = os.path.split(audio_path)[-1]
    noisy, sr = librosa.load(audio_path, sr=16000)
    noisy_pad = np.pad(noisy, hop, mode='reflect')
    noisy_pad = torch.Tensor(noisy_pad).unsqueeze(0).cuda()
    assert sr == 16000

    length = len(noisy)

    noisy_spec = torch.stft(noisy_pad, n_fft, hop, window=torch.hann_window(n_fft).cuda(), return_complex=True)
    est_spec = model(noisy_spec)

    est_audio = torch.istft(est_spec, n_fft, hop, window=torch.hann_window(n_fft).cuda())
    est_audio = torch.flatten(est_audio[:, hop:length+hop]).cpu().numpy()

    assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def evaluation():
    """
    Evaluate the model on test data and save results to a text file.
    """
    print("="*70)
    print("BSRNN Speech Enhancement - Evaluation")
    print("="*70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Test data directory: {args.test_data_dir}")
    print(f"Save enhanced audio: {args.save_enhanced_audio}")
    if args.save_enhanced_audio:
        print(f"Enhanced audio output directory: {args.enhanced_audio_dir}")
    print("="*70)
    print()

    # Initialize model
    model = BSRNN(num_channel=64, num_layer=5).cuda()

    # Load checkpoint based on model_type
    if args.model_type == 'best':
        checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint_best.pt')
        model_type = 'best'
    elif args.model_type == 'best_pesq':
        checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint_best_pesq.pt')
        model_type = 'best_pesq'
    elif args.model_type == 'last':
        checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint_last.pt')
        model_type = 'last'
    else:
        raise ValueError(f"Invalid model_type: {args.model_type}. Choose 'best', 'best_pesq', or 'last'")

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    print(f"Loading checkpoint from: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Handle both old and new checkpoint formats
    epoch_info = checkpoint.get('epoch', 'unknown')
    loss_info = f"loss: {checkpoint['gen_loss']:.6f}" if 'gen_loss' in checkpoint else ""
    pesq_info = f"PESQ: {checkpoint['pesq_score']:.4f}" if 'pesq_score' in checkpoint else ""

    info_parts = [f"Loaded model from epoch {epoch_info}"]
    if loss_info:
        info_parts.append(loss_info)
    if pesq_info:
        info_parts.append(pesq_info)

    print(", ".join(info_parts))

    model.eval()

    # Create directory for saving enhanced audio files
    if args.save_enhanced_audio:
        os.makedirs(args.enhanced_audio_dir, exist_ok=True)
        print(f"Enhanced audio files will be saved to: {args.enhanced_audio_dir}\n")

    # Prepare test directories
    noisy_dir = os.path.join(args.test_data_dir, 'noisy')
    clean_dir = os.path.join(args.test_data_dir, 'clean')

    # Get audio file list
    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)

    print(f"Evaluating on {num} audio files...\n")

    # Initialize metrics
    metrics_total = np.zeros(6)

    # Process each audio file
    for audio in tqdm(audio_list, desc="Processing"):
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)

        # Enhance audio
        est_audio, length = enhance_one_track(
            model, noisy_path, args.enhanced_audio_dir,
            16000*2, args.n_fft, args.hop_length, args.save_enhanced_audio
        )

        # Load reference audio
        noisy_audio, sr = librosa.load(noisy_path, sr=16000)
        clean_audio, sr = librosa.load(clean_path, sr=16000)
        assert sr == 16000

        # Compute metrics
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    # Calculate average metrics
    metrics_avg = metrics_total / num

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f'PESQ: {metrics_avg[0]:.4f}')
    print(f'CSIG: {metrics_avg[1]:.4f}')
    print(f'CBAK: {metrics_avg[2]:.4f}')
    print(f'COVL: {metrics_avg[3]:.4f}')
    print(f'SSNR: {metrics_avg[4]:.4f}')
    print(f'STOI: {metrics_avg[5]:.4f}')
    print("="*70)

    # Save results to text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{model_type}_{timestamp}.txt"
    results_path = os.path.join(args.checkpoint_dir, results_filename)

    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BSRNN Speech Enhancement - Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {checkpoint_file}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'unknown')}\n")

        # Handle both old and new checkpoint formats
        if 'gen_loss' in checkpoint:
            f.write(f"Training loss: {checkpoint['gen_loss']:.6f}\n")
        elif 'best_loss' in checkpoint:
            f.write(f"Best loss: {checkpoint['best_loss']:.6f}\n")

        if 'pesq_score' in checkpoint:
            f.write(f"Training PESQ: {checkpoint['pesq_score']:.4f}\n")
        if 'best_pesq' in checkpoint:
            f.write(f"Best PESQ: {checkpoint['best_pesq']:.4f}\n")

        f.write(f"Test data directory: {args.test_data_dir}\n")
        f.write(f"Number of test files: {num}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"PESQ: {metrics_avg[0]:.4f}\n")
        f.write(f"CSIG: {metrics_avg[1]:.4f}\n")
        f.write(f"CBAK: {metrics_avg[2]:.4f}\n")
        f.write(f"COVL: {metrics_avg[3]:.4f}\n")
        f.write(f"SSNR: {metrics_avg[4]:.4f}\n")
        f.write(f"STOI: {metrics_avg[5]:.4f}\n")
        f.write("="*70 + "\n")

    print(f"\nResults saved to: {results_path}\n")

    return metrics_avg


if __name__ == '__main__':
    evaluation()
