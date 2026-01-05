import os
import dataloader
import torch
import logging
from train_improved import ImprovedTrainer, Config, args

# ============================================
# RESUME TRAINING CONFIGURATION
# ============================================
class ResumeConfig(Config):
    # Specify which checkpoint to resume from
    # Options: 'last', 'best_loss', 'best_pesq', 'best_stoi', 'best_composite'
    resume_from = 'last'  # Usually resume from last checkpoint

    # Override epochs if you want to train longer
    # Set to None to keep the original epochs setting
    new_total_epochs = None  # e.g., 500 to train to 500 total epochs

args = ResumeConfig()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def resume_training():
    """
    Resume training from a saved checkpoint.
    """
    logging.info("="*70)
    logging.info("RESUMING TRAINING")
    logging.info("="*70)

    # Determine checkpoint file
    checkpoint_files = {
        'best_loss': 'checkpoint_best_loss.pt',
        'best_pesq': 'checkpoint_best_pesq.pt',
        'best_stoi': 'checkpoint_best_stoi.pt',
        'best_composite': 'checkpoint_best_composite.pt',
        'last': 'checkpoint_last.pt'
    }

    if args.resume_from not in checkpoint_files:
        raise ValueError(f"Invalid resume_from: {args.resume_from}. "
                        f"Choose from {list(checkpoint_files.keys())}")

    checkpoint_path = os.path.join(args.save_model_dir, checkpoint_files[args.resume_from])

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    logging.info(f"Resuming from epoch {checkpoint['epoch']}")
    logging.info(f"  Previous validation loss: {checkpoint.get('gen_loss', 'N/A')}")
    logging.info(f"  Previous PESQ: {checkpoint.get('pesq_score', 'N/A')}")
    logging.info(f"  Previous STOI: {checkpoint.get('stoi_score', 'N/A')}")

    # Load data
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 4, args.cut_len)

    # Create trainer
    trainer = ImprovedTrainer(train_ds, test_ds)

    # Restore model state
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])

    # Restore training state
    trainer.start_epoch = checkpoint['epoch'] + 1
    trainer.best_loss = checkpoint.get('best_loss', float('inf'))
    trainer.best_pesq = checkpoint.get('best_pesq', 0.0)
    trainer.best_stoi = checkpoint.get('best_stoi', 0.0)
    trainer.best_composite = checkpoint.get('best_composite', -float('inf'))

    # Update total epochs if specified
    if args.new_total_epochs is not None:
        original_epochs = args.epochs
        args.epochs = args.new_total_epochs
        logging.info(f"Extended training from {original_epochs} to {args.epochs} epochs")

        # Recalculate discriminator start epoch based on new total
        trainer.disc_start_epoch = int(args.epochs * args.disc_start_ratio)
        logging.info(f"Discriminator start epoch recalculated: {trainer.disc_start_epoch}")

    logging.info(f"Will train from epoch {trainer.start_epoch} to {args.epochs}")
    logging.info(f"Best loss so far: {trainer.best_loss:.6f}")
    logging.info(f"Best PESQ so far: {trainer.best_pesq:.4f}")
    logging.info(f"Best STOI so far: {trainer.best_stoi:.4f}")
    logging.info(f"Best composite so far: {trainer.best_composite:.4f}")
    logging.info("="*70)

    # Resume training
    trainer.train()


if __name__ == '__main__':
    resume_training()