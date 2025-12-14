import os
import dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchinfo import summary
from natsort import natsorted
import librosa
import numpy as np
from module import *
from utils import batch_pesq  # ✅ FIXED: Added missing import
from pystoi import stoi as stoi_metric  # pip install pystoi

# ============================================
# IMPROVED CONFIGURATION FOR BALANCED TRAINING
# ============================================
class Config:
    # Training hyperparameters
    epochs = 250
    batch_size = 6
    log_interval = 500
    decay_epoch = 10
    init_lr = 1e-3
    cut_len = int(16000 * 2)  # 2 seconds at 16kHz

    # IMPROVED LOSS WEIGHTS - Better balanced
    # Old: [0.5, 0.5, 1.0] caused discriminator dominance
    # New: Reduced discriminator weight, added time-domain loss
    loss_weights = {
        'ri': 0.45,           # Real-Imaginary loss (reduced from 0.5)
        'mag': 0.45,          # Magnitude loss (reduced from 0.5)
        'time': 0.10,         # Time-domain loss (NEW - helps STOI)
        'disc': 0.30          # Discriminator loss (reduced from 1.0)
    }

    # Discriminator configuration
    disc_start_ratio = 0.60      # Start discriminator at 60% of training (was 50%)
    disc_warmup_epochs = 20      # Gradual warmup over 20 epochs

    # Checkpoint selection weights for composite score
    # Adjust these based on your priorities
    composite_weights = {
        'pesq': 0.50,      # Primary metric
        'stoi': 0.30,      # Intelligibility
        'loss': -0.20      # Negative because lower is better
    }

    # Server paths - MODIFY THESE FOR YOUR SERVER
    data_dir = '/gdata/fewahab/data/VoicebanK-demand-16K'
    save_model_dir = '/ghome/fewahab/My_5th_pap/Trinet-improved/ckpt'

    # Validation STOI computation (set to True to enable)
    compute_stoi_validation = True  # Computes STOI during validation

args = Config()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ImprovedTrainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 512
        self.hop = 128
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.model = BSRNN(num_channel=64, num_layer=5).cuda()
        self.discriminator = Discriminator(ndf=16).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=args.init_lr)

        # Training state - track multiple "best" checkpoints
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.best_pesq = 0.0
        self.best_stoi = 0.0
        self.best_composite = -float('inf')

        # Discriminator starting epoch
        self.disc_start_epoch = int(args.epochs * args.disc_start_ratio)

        logging.info(f"Discriminator will start at epoch {self.disc_start_epoch}")
        logging.info(f"Discriminator warmup over {args.disc_warmup_epochs} epochs")
        logging.info(f"Loss weights: RI={args.loss_weights['ri']}, Mag={args.loss_weights['mag']}, "
                    f"Time={args.loss_weights['time']}, Disc={args.loss_weights['disc']}")

    # ----------------- CHECKPOINT SAVING -----------------
    def save_checkpoint(self, epoch, gen_loss, pesq_score, stoi_score, composite_score,
                       is_best_loss=False, is_best_pesq=False, is_best_stoi=False, is_best_composite=False):
        """
        Save multiple checkpoint types:
          - checkpoint_last.pt           : always (most recent)
          - checkpoint_best_loss.pt      : best validation loss
          - checkpoint_best_pesq.pt      : best PESQ score
          - checkpoint_best_stoi.pt      : best STOI score
          - checkpoint_best_composite.pt : best composite score (RECOMMENDED)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'best_loss': self.best_loss,
            'best_pesq': self.best_pesq,
            'best_stoi': self.best_stoi,
            'best_composite': self.best_composite,
            'gen_loss': gen_loss,
            'pesq_score': pesq_score,
            'stoi_score': stoi_score,
            'composite_score': composite_score,
            'config': {
                'loss_weights': args.loss_weights,
                'disc_start_epoch': self.disc_start_epoch,
                'epochs': args.epochs
            }
        }

        os.makedirs(args.save_model_dir, exist_ok=True)

        # Always save latest
        latest_path = os.path.join(args.save_model_dir, 'checkpoint_last.pt')
        torch.save(checkpoint, latest_path)

        # Save best loss
        if is_best_loss:
            best_path = os.path.join(args.save_model_dir, 'checkpoint_best_loss.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"✅ New best LOSS model saved: epoch {epoch}, loss {gen_loss:.6f}")

        # Save best PESQ
        if is_best_pesq:
            best_pesq_path = os.path.join(args.save_model_dir, 'checkpoint_best_pesq.pt')
            torch.save(checkpoint, best_pesq_path)
            logging.info(f"✅ New best PESQ model saved: epoch {epoch}, PESQ {pesq_score:.4f}")

        # Save best STOI
        if is_best_stoi:
            best_stoi_path = os.path.join(args.save_model_dir, 'checkpoint_best_stoi.pt')
            torch.save(checkpoint, best_stoi_path)
            logging.info(f"✅ New best STOI model saved: epoch {epoch}, STOI {stoi_score:.4f}")

        # Save best composite (RECOMMENDED)
        if is_best_composite:
            best_composite_path = os.path.join(args.save_model_dir, 'checkpoint_best_composite.pt')
            torch.save(checkpoint, best_composite_path)
            logging.info(f"✅ New best COMPOSITE model saved: epoch {epoch}, "
                        f"composite {composite_score:.4f} "
                        f"(PESQ {pesq_score:.4f}, STOI {stoi_score:.4f}, loss {gen_loss:.6f})")

    def get_discriminator_weight(self, epoch):
        """
        Calculate discriminator weight with gradual warmup.

        Returns:
            float: Discriminator weight (0.0 to args.loss_weights['disc'])
        """
        if epoch < self.disc_start_epoch:
            return 0.0
        elif epoch < self.disc_start_epoch + args.disc_warmup_epochs:
            # Linear warmup
            progress = (epoch - self.disc_start_epoch) / args.disc_warmup_epochs
            return args.loss_weights['disc'] * progress
        else:
            return args.loss_weights['disc']

    # ----------------- TRAIN STEP -----------------
    def train_step(self, batch, use_disc, disc_weight):
        """
        Training step with improved loss function.

        Improvements:
        1. Time-domain loss for better STOI
        2. Gradual discriminator weight
        3. Better loss balancing
        """
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        self.optimizer.zero_grad()

        # STFT
        noisy_spec = torch.stft(
            noisy, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True, return_complex=True
        )
        clean_spec = torch.stft(
            clean, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True, return_complex=True
        )

        # Model forward
        est_spec = self.model(noisy_spec)

        # Power compression (0.3 exponent)
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** 0.3
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** 0.3
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** 0.3

        # Loss 1: Real-Imaginary loss
        mae_loss = nn.L1Loss()
        loss_ri = mae_loss(est_spec, clean_spec)

        # Loss 2: Magnitude loss
        loss_mag = mae_loss(est_mag, clean_mag)

        # Loss 3: Time-domain loss (NEW - helps STOI preservation)
        est_audio = torch.istft(
            est_spec, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True
        )
        loss_time = mae_loss(est_audio, clean)

        # Loss 4: Discriminator loss (with gradual weight)
        if use_disc and disc_weight > 0:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
        else:
            gen_loss_GAN = torch.tensor(0.0).cuda()

        # COMBINED LOSS with improved weights
        loss = (
            args.loss_weights['ri'] * loss_ri +
            args.loss_weights['mag'] * loss_mag +
            args.loss_weights['time'] * loss_time +
            disc_weight * gen_loss_GAN
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
        self.optimizer.step()

        # Compute PESQ for discriminator training
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        # Denormalized PESQ for logging
        pesq_raw = None
        if pesq_score is not None:
            pesq_raw = (pesq_score.mean().item() * 5) - 0.5

        # Discriminator training
        if use_disc and pesq_score is not None and pesq_score_n is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)

            discrim_loss_metric = (
                F.mse_loss(predict_max_metric.flatten(), one_labels.float()) +
                F.mse_loss(predict_enhance_metric.flatten(), pesq_score) +
                F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
            )

            discrim_loss_metric.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_raw, loss_ri.item(), loss_mag.item(), loss_time.item()

    # ----------------- TEST STEP -----------------
    @torch.no_grad()
    def test_step(self, batch, use_disc, disc_weight, compute_stoi=False):
        """
        Validation step with optional STOI computation.
        """
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        # STFT
        noisy_spec = torch.stft(
            noisy, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True, return_complex=True
        )
        clean_spec = torch.stft(
            clean, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True, return_complex=True
        )

        # Model forward
        est_spec = self.model(noisy_spec)

        # Power compression
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** 0.3
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** 0.3
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** 0.3

        mae_loss = nn.L1Loss()
        loss_ri = mae_loss(est_spec, clean_spec)
        loss_mag = mae_loss(est_mag, clean_mag)

        # Time-domain reconstruction
        est_audio = torch.istft(
            est_spec, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True
        )
        loss_time = mae_loss(est_audio, clean)

        # Discriminator loss
        if use_disc and disc_weight > 0:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
        else:
            gen_loss_GAN = torch.tensor(0.0).cuda()

        # Combined loss
        loss = (
            args.loss_weights['ri'] * loss_ri +
            args.loss_weights['mag'] * loss_mag +
            args.loss_weights['time'] * loss_time +
            disc_weight * gen_loss_GAN
        )

        # Compute metrics
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        pesq_raw = None
        if pesq_score is not None:
            pesq_raw = (pesq_score.mean().item() * 5) - 0.5

        # STOI computation (optional, slower)
        stoi_score = None
        if compute_stoi and pesq_raw is not None:
            try:
                stoi_scores = []
                for clean_np, est_np in zip(clean_audio_list, est_audio_list):
                    stoi_val = stoi_metric(clean_np, est_np, 16000, extended=False)
                    stoi_scores.append(stoi_val)
                stoi_score = np.mean(stoi_scores)
            except Exception as e:
                logging.warning(f"STOI computation failed: {e}")
                stoi_score = None

        # Discriminator loss
        if use_disc and pesq_score is not None and pesq_score_n is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)

            discrim_loss_metric = (
                F.mse_loss(predict_max_metric.flatten(), one_labels) +
                F.mse_loss(predict_enhance_metric.flatten(), pesq_score) +
                F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
            )
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_raw, stoi_score

    # ----------------- VALIDATION -----------------
    def test(self, epoch, use_disc, disc_weight):
        """
        Validation with multi-metric tracking.
        """
        self.model.eval()
        self.discriminator.eval()

        gen_loss_total = 0.
        disc_loss_total = 0.
        pesq_total = 0.
        pesq_count = 0
        stoi_total = 0.
        stoi_count = 0

        # Only compute STOI periodically (every 10 epochs) to save time
        compute_stoi = args.compute_stoi_validation and (epoch % 10 == 0 or epoch == args.epochs - 1)

        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss, pesq_raw, stoi_raw = self.test_step(batch, use_disc, disc_weight, compute_stoi)

            gen_loss_total += loss
            disc_loss_total += disc_loss

            if pesq_raw is not None:
                pesq_total += pesq_raw
                pesq_count += 1

            if stoi_raw is not None:
                stoi_total += stoi_raw
                stoi_count += 1

        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        pesq_avg = pesq_total / pesq_count if pesq_count > 0 else 0
        stoi_avg = stoi_total / stoi_count if stoi_count > 0 else 0

        # Compute composite score
        # PESQ normalized to ~0-1 range (assuming 1.0-4.5 range)
        pesq_norm = (pesq_avg - 1.0) / 3.5 if pesq_avg > 0 else 0
        composite_score = (
            args.composite_weights['pesq'] * pesq_norm +
            args.composite_weights['stoi'] * stoi_avg +
            args.composite_weights['loss'] * gen_loss_avg
        )

        template = 'TEST - Loss: {:.4f}, Disc Loss: {:.4f}, PESQ: {:.4f}'
        if stoi_avg > 0:
            template += ', STOI: {:.4f}, Composite: {:.4f}'
            logging.info(template.format(gen_loss_avg, disc_loss_avg, pesq_avg, stoi_avg, composite_score))
        else:
            logging.info(template.format(gen_loss_avg, disc_loss_avg, pesq_avg))

        return gen_loss_avg, pesq_avg, stoi_avg, composite_score

    # ----------------- TRAINING LOOP -----------------
    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.98
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.98
        )

        logging.info("="*70)
        logging.info("IMPROVED TRAINING - Multi-Metric Optimization")
        logging.info("="*70)
        logging.info(f"Total epochs: {args.epochs}")
        logging.info(f"Discriminator starts at: epoch {self.disc_start_epoch}")
        logging.info(f"Discriminator warmup: {args.disc_warmup_epochs} epochs")
        logging.info(f"Loss weights: {args.loss_weights}")
        logging.info("="*70)

        for epoch in range(self.start_epoch, args.epochs):
            self.model.train()
            self.discriminator.train()

            loss_total = 0
            loss_gan = 0
            loss_ri_total = 0
            loss_mag_total = 0
            loss_time_total = 0
            pesq_total = 0
            pesq_count = 0

            # Determine discriminator usage and weight
            use_disc = epoch >= self.disc_start_epoch
            disc_weight = self.get_discriminator_weight(epoch)

            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss, pesq_raw, loss_ri, loss_mag, loss_time = self.train_step(
                    batch, use_disc, disc_weight
                )

                loss_total += loss
                loss_gan += disc_loss
                loss_ri_total += loss_ri
                loss_mag_total += loss_mag
                loss_time_total += loss_time

                if pesq_raw is not None:
                    pesq_total += pesq_raw
                    pesq_count += 1

                if (step % args.log_interval) == 0:
                    pesq_avg = pesq_total / pesq_count if pesq_count > 0 else 0
                    template = ('Epoch {}, Step {}, Loss: {:.4f} '
                              '(RI: {:.4f}, Mag: {:.4f}, Time: {:.4f}, Disc: {:.4f}), '
                              'Disc Loss: {:.4f}, PESQ: {:.4f}, Disc Weight: {:.2f}')
                    logging.info(template.format(
                        epoch, step,
                        loss_total / step,
                        loss_ri_total / step,
                        loss_mag_total / step,
                        loss_time_total / step,
                        loss_gan / step * disc_weight if use_disc else 0,
                        loss_gan / step,
                        pesq_avg,
                        disc_weight
                    ))

            # Validation
            gen_loss, pesq_score, stoi_score, composite_score = self.test(epoch, use_disc, disc_weight)

            # Check for best checkpoints
            is_best_loss = gen_loss < self.best_loss
            if is_best_loss:
                self.best_loss = gen_loss

            is_best_pesq = pesq_score > self.best_pesq
            if is_best_pesq:
                self.best_pesq = pesq_score

            is_best_stoi = stoi_score > self.best_stoi and stoi_score > 0
            if is_best_stoi:
                self.best_stoi = stoi_score

            is_best_composite = composite_score > self.best_composite
            if is_best_composite:
                self.best_composite = composite_score

            # Save checkpoints
            self.save_checkpoint(
                epoch, gen_loss, pesq_score, stoi_score, composite_score,
                is_best_loss=is_best_loss,
                is_best_pesq=is_best_pesq,
                is_best_stoi=is_best_stoi,
                is_best_composite=is_best_composite
            )

            scheduler_G.step()
            scheduler_D.step()

        logging.info("="*70)
        logging.info("TRAINING COMPLETED")
        logging.info("="*70)
        logging.info(f"Best validation loss: {self.best_loss:.6f}")
        logging.info(f"Best PESQ: {self.best_pesq:.4f}")
        logging.info(f"Best STOI: {self.best_stoi:.4f}")
        logging.info(f"Best composite: {self.best_composite:.4f}")
        logging.info("="*70)
        logging.info("RECOMMENDATION: Use 'checkpoint_best_composite.pt' for evaluation")
        logging.info("="*70)


def main():
    logging.info("Training configuration:")
    logging.info(
        f"epochs={args.epochs}, batch_size={args.batch_size}, init_lr={args.init_lr}, "
        f"data_dir={args.data_dir}, save_model_dir={args.save_model_dir}"
    )
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    logging.info(f"Available GPUs: {available_gpus}")

    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 4, args.cut_len)
    trainer = ImprovedTrainer(train_ds, test_ds)
    trainer.train()


if __name__ == '__main__':
    main()
