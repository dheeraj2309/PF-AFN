from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--warp_checkpoint', type=str, default='checkpoints/PFAFN/warp_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='checkpoints/PFAFN/gen_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--unpaired', action='store_true', help='if enables, uses unpaired data from dataset')
        self.parser.add_argument('--num_test_samples', type=int, default=-1, help='Number of random images to test. -1 or 0 means all.')
        self.parser.add_argument('--random_seed', type=int, default=42, help='Seed for random subset selection for reproducibility.')
        self.isTrain = False
