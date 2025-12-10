#!/usr/bin/env python
# coding: utf-8

import numpy as np
from module import *
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import argparse
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf

@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=512, hop=128, save_tracks=False):
    name = os.path.split(audio_path)[-1]
    noisy, sr = librosa.load(audio_path,sr=16000)
    noisy_pad = np.pad(noisy,hop, mode='reflect')
    noisy_pad = torch.Tensor(noisy_pad).unsqueeze(0).cuda()
    assert sr == 16000
    
    length = len(noisy)
    
    noisy_spec = torch.stft(noisy_pad, n_fft, hop, window=torch.hann_window(n_fft).cuda(),return_complex=True)
    est_spec = model(noisy_spec)
        
    est_audio = torch.istft(est_spec, n_fft, hop, window=torch.hann_window(n_fft).cuda())
    est_audio = torch.flatten(est_audio[:,hop:length+hop]).cpu().numpy()
    
    assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir, use_best=True):
    """
    Evaluate the model on test data.

    Args:
        model_path: Directory containing the saved checkpoints
        noisy_dir: Directory with noisy test audio files
        clean_dir: Directory with clean test audio files
        save_tracks: Whether to save enhanced audio files
        saved_dir: Directory where enhanced audio files will be saved
        use_best: If True, load best_model.pth; if False, load checkpoint_latest.pth
    """
    n_fft = 512
    model = BSRNN(num_channel=64, num_layer=5).cuda()

    # Load checkpoint (best or latest)
    if use_best:
        checkpoint_file = os.path.join(model_path, 'best_model.pth')
    else:
        checkpoint_file = os.path.join(model_path, 'checkpoint_latest.pth')

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    print(f"Loading checkpoint from: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['gen_loss']:.6f}")
    model.eval()

    # Create directory for saving enhanced audio files
    if save_tracks:
        os.makedirs(saved_dir, exist_ok=True)
        print(f"Enhanced audio files will be saved to: {saved_dir}")

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    for audio in tqdm(audio_list):
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        est_audio, length = enhance_one_track(model, noisy_path, saved_dir, 16000*2, n_fft, n_fft//4, save_tracks)
        noisy_audio, sr = librosa.load(noisy_path,sr=16000)
        clean_audio, sr = librosa.load(clean_path,sr=16000)
        assert sr == 16000        
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics
    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 'covl: ',
          metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


parser = argparse.ArgumentParser(description='Evaluate BSRNN speech enhancement model')

# Model checkpoint configuration
parser.add_argument("--model_path", type=str,
                    default='/ghome/fewahab/My_5th_pap/Ab4-BSRNN/B1/ckpt',
                    help="Directory where model checkpoints are saved (contains best_model.pth and checkpoint_latest.pth)")
parser.add_argument("--use_best", action='store_true', default=True,
                    help="Use best_model.pth (default: True). Use --no-use_best for checkpoint_latest.pth")
parser.add_argument("--no-use_best", dest='use_best', action='store_false',
                    help="Use checkpoint_latest.pth instead of best_model.pth")

# Test data configuration
parser.add_argument("--test_dir", type=str,
                    default='/gdata/fewahab/data/VoicebanK-demand-16K/test',
                    help="Test directory containing 'noisy' and 'clean' subdirectories")

# Enhanced audio output configuration
parser.add_argument("--save_tracks", action='store_true', default=True,
                    help="Save enhanced audio tracks (default: True)")
parser.add_argument("--no-save_tracks", dest='save_tracks', action='store_false',
                    help="Skip saving enhanced audio tracks")
parser.add_argument("--save_dir", type=str,
                    default='/ghome/fewahab/My_5th_pap/Ab4-BSRNN/B1/enhanced_audio',
                    help="Directory where enhanced audio files will be saved")

args = parser.parse_args()


if __name__ == '__main__':
    print("="*70)
    print("BSRNN Speech Enhancement - Evaluation")
    print("="*70)
    print(f"Model checkpoint directory: {args.model_path}")
    print(f"Using {'BEST' if args.use_best else 'LATEST'} model checkpoint")
    print(f"Test data directory: {args.test_dir}")
    print(f"Save enhanced audio: {args.save_tracks}")
    if args.save_tracks:
        print(f"Enhanced audio output directory: {args.save_dir}")
    print("="*70)
    print()

    noisy_dir = os.path.join(args.test_dir, 'noisy')
    clean_dir = os.path.join(args.test_dir, 'clean')

    evaluation(args.model_path, noisy_dir, clean_dir, args.save_tracks, args.save_dir, args.use_best)

