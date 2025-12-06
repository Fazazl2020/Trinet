import os
import dataloader
import torch
import torch.nn.functional as F
import logging
from torchinfo import summary
from natsort import natsorted
import librosa
# import matplotlib.pyplot as plt  # Not used - removed for server compatibility
import numpy as np
from tqdm import tqdm
from module import *

# ============================================
# CONFIGURATION - HARDCODED FOR SERVER
# ============================================
class Config:
    # Training hyperparameters
    epochs = 120
    batch_size = 6
    log_interval = 500
    decay_epoch = 10
    init_lr = 1e-3
    cut_len = int(16000 * 2)  # 2 seconds at 16kHz
    loss_weights = [0.5, 0.5, 1]  # [RI, magnitude, Metric Disc]

    # Server paths - MODIFY THESE FOR YOUR SERVER
    data_dir = '/gdata/fewahab/data/VoicebanK-demand-16K'
    save_model_dir =  '/ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model'

    # ========================================
    # RESUME TRAINING CONFIGURATION (SIMPLIFIED!)
    # ========================================
    # To resume training:
    # 1. Set resume_from_checkpoint to the checkpoint directory path
    # 2. Set which checkpoint to load: 'best' or 'last'
    # 3. Optionally increase epochs for more training
    #
    # The code will automatically:
    # - Find the checkpoint file (checkpoint_best.pt or checkpoint_last.pt)
    # - Load models, optimizers, schedulers, epoch number
    # - Resume training seamlessly
    #
    # Example:
    #   resume_from_checkpoint = '/ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model'
    #   load_checkpoint_type = 'best'  # or 'last'
    #   epochs = 200  # Train for 80 more epochs (if previously trained 120)

    resume_from_checkpoint = None  # Path to checkpoint directory (or None to start fresh)
    load_checkpoint_type = 'best'  # 'best' or 'last'

args = Config()
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 512
        self.hop = 128
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.model = BSRNN(num_channel=64, num_layer=5).cuda()
#         summary(self.model, [(1, 257, args.cut_len//self.hop+1, 2)])
        self.discriminator = Discriminator(ndf=16).cuda()
# #         summary(self.discriminator, [(1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1),
# #                                      (1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1)])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=args.init_lr)

        # Checkpoint tracking
        self.start_epoch = 0
        self.best_val_loss = float('inf')  # Track best validation loss
        self.loaded_scheduler_states = None  # Store loaded scheduler states

        # Resume from checkpoint if specified
        if args.resume_from_checkpoint is not None:
            self.loaded_scheduler_states = self._load_checkpoint()

    def _load_checkpoint(self):
        """
        Load checkpoint from directory (AUTO-DETECTS checkpoint file)

        Returns:
            scheduler_G, scheduler_D (if saved in checkpoint, else None)
        """
        logging.info('='*70)
        logging.info('RESUMING FROM CHECKPOINT')
        logging.info('='*70)

        # Determine checkpoint filename
        checkpoint_name = f'checkpoint_{args.load_checkpoint_type}.pt'
        checkpoint_path = os.path.join(args.resume_from_checkpoint, checkpoint_name)

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f'\n'
                f'ERROR: Checkpoint not found!\n'
                f'Looking for: {checkpoint_path}\n'
                f'\n'
                f'Make sure:\n'
                f'  1. The checkpoint directory exists: {args.resume_from_checkpoint}\n'
                f'  2. The checkpoint file exists: {checkpoint_name}\n'
                f'  3. You specified the correct load_checkpoint_type: "{args.load_checkpoint_type}"\n'
                f'\n'
                f'Available checkpoint types: "best" or "last"\n'
            )

        logging.info(f'Loading checkpoint from: {checkpoint_path}')

        try:
            checkpoint = torch.load(checkpoint_path)
        except Exception as e:
            raise RuntimeError(
                f'ERROR: Failed to load checkpoint!\n'
                f'File: {checkpoint_path}\n'
                f'Error: {str(e)}\n'
            )

        # Load model states
        logging.info('Loading model states...')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        logging.info('✓ Models loaded successfully')

        # Load optimizer states (CRITICAL for resume!)
        logging.info('Loading optimizer states...')
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        logging.info('✓ Optimizers loaded successfully')

        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Load scheduler states (if available)
        scheduler_G_state = checkpoint.get('scheduler_G_state_dict', None)
        scheduler_D_state = checkpoint.get('scheduler_D_state_dict', None)

        # Validation: make sure we're not resuming beyond target epochs
        if self.start_epoch >= args.epochs:
            logging.warning(
                f'\n'
                f'WARNING: Resume epoch ({self.start_epoch}) >= target epochs ({args.epochs})\n'
                f'You completed {self.start_epoch} epochs, but epochs={args.epochs}\n'
                f'\n'
                f'To train more epochs, increase "epochs" in Config.\n'
                f'Example: epochs = {self.start_epoch + 80}  # Train 80 more epochs\n'
            )

        # Print summary
        logging.info('='*70)
        logging.info('RESUME SUMMARY:')
        logging.info(f'  Checkpoint type: {args.load_checkpoint_type}')
        logging.info(f'  Completed epochs: 0-{checkpoint["epoch"]} ({checkpoint["epoch"] + 1} epochs)')
        logging.info(f'  Resuming from: epoch {self.start_epoch}')
        logging.info(f'  Target epochs: {args.epochs}')
        logging.info(f'  Remaining epochs: {max(0, args.epochs - self.start_epoch)}')
        logging.info(f'  Best validation loss: {self.best_val_loss:.6f}')
        logging.info(f'  Last validation loss: {checkpoint["val_loss"]:.6f}')
        logging.info('='*70 + '\n')

        return scheduler_G_state, scheduler_D_state

    def train_step(self, batch, use_disc):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        self.optimizer.zero_grad()
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)

        est_spec = self.model(noisy_spec)
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** (0.3)
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** (0.3)
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** (0.3)

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec,clean_spec)

        if use_disc is False:
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * gen_loss_GAN

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
        self.optimizer.step()

        est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                           onesided =True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        # Store PESQ score for logging (denormalize from [0,1] back to [-0.5, 4.5] range)
        pesq_raw = None
        if pesq_score is not None:
            pesq_raw = (pesq_score.mean().item() * 5) - 0.5

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None and pesq_score_n is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels.float()) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)

            discrim_loss_metric.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_raw

    @torch.no_grad()
    def test_step(self, batch,use_disc):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)

        est_spec = self.model(noisy_spec)
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** (0.3)
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** (0.3)
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** (0.3)

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        if use_disc is False:
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * gen_loss_GAN

        est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                           onesided =True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        # Store PESQ score for logging (denormalize from [0,1] back to [-0.5, 4.5] range)
        pesq_raw = None
        if pesq_score is not None:
            pesq_raw = (pesq_score.mean().item() * 5) - 0.5

        if pesq_score is not None and pesq_score_n is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_raw

    def test(self,use_disc):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        pesq_total = 0.
        pesq_count = 0
        for idx, batch in enumerate(tqdm(self.test_ds,
                                              ncols=100,
                                              mininterval=10.0,
                                              desc='Validation')):
            step = idx + 1
            loss, disc_loss, pesq_raw = self.test_step(batch,use_disc)
            gen_loss_total += loss
            disc_loss_total += disc_loss
            if pesq_raw is not None:
                pesq_total += pesq_raw
                pesq_count += 1
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        pesq_avg = pesq_total / pesq_count if pesq_count > 0 else 0

        template = 'TEST - Generator loss: {:.4f}, Discriminator loss: {:.4f}, PESQ: {:.4f}'
        logging.info(template.format(gen_loss_avg, disc_loss_avg, pesq_avg))

        return gen_loss_avg

    def _save_checkpoint(self, epoch, val_loss, scheduler_G, scheduler_D, is_best=False):
        """
        Save checkpoint with ALL training state

        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            scheduler_G: Generator scheduler
            scheduler_D: Discriminator scheduler
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }

        # Ensure save directory exists
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)

        # Always save as "last" checkpoint
        last_path = os.path.join(args.save_model_dir, 'checkpoint_last.pt')
        torch.save(checkpoint, last_path)
        logging.info(f'✓ Saved checkpoint_last.pt (epoch {epoch}, val_loss: {val_loss:.6f})')

        # Save as "best" if this is the best so far
        if is_best:
            best_path = os.path.join(args.save_model_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            logging.info(f'🏆 NEW BEST! Saved checkpoint_best.pt (val_loss: {val_loss:.6f})')

    def train(self):
        # Create schedulers
        scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_epoch, gamma=0.98)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=args.decay_epoch, gamma=0.98)

        # Load scheduler states if resuming
        if self.loaded_scheduler_states is not None:
            scheduler_G_state, scheduler_D_state = self.loaded_scheduler_states
            if scheduler_G_state is not None:
                scheduler_G.load_state_dict(scheduler_G_state)
                scheduler_D.load_state_dict(scheduler_D_state)
                logging.info(f'✓ Scheduler states loaded')
                logging.info(f'  Current learning rate: {scheduler_G.get_last_lr()[0]:.6f}')
            else:
                # Old checkpoint without scheduler states - manually adjust
                logging.info('⚠ Scheduler states not found in checkpoint, manually adjusting...')
                for _ in range(self.start_epoch):
                    scheduler_G.step()
                    scheduler_D.step()
                logging.info(f'  Scheduler adjusted: learning rate = {scheduler_G.get_last_lr()[0]:.6f}')

        # Training loop
        for epoch in range(self.start_epoch, args.epochs):
            self.model.train()
            self.discriminator.train()

            loss_total = 0
            loss_gan = 0
            pesq_total = 0
            pesq_count = 0

            # Use discriminator after half the epochs
            if epoch >= (args.epochs/2):
                use_disc = True
            else:
                use_disc = False

            # Training
            for idx, batch in enumerate(tqdm(self.train_ds,
                                              ncols=100,
                                              mininterval=10.0,
                                              desc=f'Epoch {epoch}')):
                step = idx + 1
                loss, disc_loss, pesq_raw = self.train_step(batch, use_disc)

                loss_total = loss_total + loss
                loss_gan = loss_gan + disc_loss
                if pesq_raw is not None:
                    pesq_total += pesq_raw
                    pesq_count += 1

                if (step % args.log_interval) == 0:
                    pesq_avg = pesq_total/pesq_count if pesq_count > 0 else 0
                    template = 'Epoch {}, Step {}, loss: {:.4f}, disc_loss: {:.4f}, PESQ: {:.4f}'
                    logging.info(template.format(epoch, step, loss_total/step, loss_gan/step, pesq_avg))

            # Validation
            val_loss = self.test(use_disc)

            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint (saves both "last" and "best" if applicable)
            self._save_checkpoint(epoch, val_loss, scheduler_G, scheduler_D, is_best=is_best)

            # Step schedulers
            scheduler_G.step()
            scheduler_D.step()

def main():
    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 4, args.cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()

if __name__ == '__main__':
    main()
