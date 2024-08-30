
class Args_:
    def __init__(self):
        self._id: str = ''
        self._num_mix: int = 2
        self._arch_sound_ground: str = 'vggish'
        self._arch_frame_ground: str = 'resnet18'
        self._arch_sound: str = 'unet7'
        self._arch_frame: str = 'resnet18dilated'
        self._arch_synthesizer: str = 'linear'
        self._arch_grounding: str = 'base'
        self._weights_sound_ground: str = ''
        self._weights_frame_ground: str = ''
        self._weights_sound: str = ''
        self._weights_frame: str = ''
        self._weights_synthesizer: str = ''
        self._weights_grounding: str = ''
        self._weights_maskformer: str = ''
        self._num_channels: int = 32
        self._num_frames: int = 64
        self._stride_frames: int = 1
        self._img_pool: str = 'maxpool'
        self._img_activation: str = 'sigmoid'
        self._sound_activation: str = 'no'
        self._output_activation: str = 'sigmoid'
        self._binary_mask: int = 0
        self._mask_thres: float = 0.5
        self._loss: str = 'l1'
        self._weighted_loss: int = 0
        self._log_freq: int = 1
        self._split: str = 'val'
        self._num_gpus: int = 1
        self._batch_size_per_gpu: int = 32
        self._workers: int = 16
        self._num_val: int = -1
        self._num_vis: int = 40
        self._audLen: int = 65535
        self._audRate: int = 11025
        self._stft_frame: int = 1022
        self._stft_hop: int = 256
        self._imgSize: int = 224
        self._frameRate: float = 25
        self._seed: int = 1234
        self._ckpt: str = '../data/ckpt'
        self._disp_iter: int = 20
        self._eval_epoch: int = 1
        self._in_channels: int = 32
        self._MASK_FORMER_HIDDEN_DIM: int = 32
        self._MASK_FORMER_NUM_OBJECT_QUERIES: int = 20
        self._MASK_FORMER_NHEADS: int = 8
        self._MASK_FORMER_DROPOUT: float = 0.1
        self._MASK_FORMER_DIM_FEEDFORWARD: int = 256
        self._MASK_FORMER_ENC_LAYERS: int = 0
        self._MASK_FORMER_DEC_LAYERS: int = 6
        self._SEM_SEG_HEAD_MASK_DIM: int = 32
        self._mode: str = 'train'
        self._list_train: str = 'data/train.csv'
        self._list_val: str = 'data/val.csv'
        self._dup_trainset: int = 40
        self._num_epoch: int = 100
        self._lr_frame_ground: float = 0.001
        self._lr_sound_ground: float = 0.001
        self._lr_frame: float = 0.0001
        self._lr_sound: float = 0.001
        self._lr_synthesizer: float = 0.001
        self._lr_grounding: float = 0.001
        self._lr_maskformer: float = 0.0001
        self._lr_steps: int = [40, 60]
        self._beta1: float = 0.9
        self._weight_decay: float = 0.0001
        self._weight_decay_maskformer: float = 0.0001
        self._lr_drop_maskformer: int = 60
        self._lr_drop_maskformer_1: int = 60
        self._lr_drop_maskformer_2: int = 80
        
    
    @property
    def id(self) -> str:
        """
        ``` yaml
        name: id
        type: str
        default: ''
        choices: null
        help: a name for identifying the model
        ```
        """
        return self._id
    
    @property
    def num_mix(self) -> int:
        """
        ``` yaml
        name: num_mix
        type: int
        default: 2
        choices: null
        help: number of sounds to mix
        ```
        """
        return self._num_mix
    
    @property
    def arch_sound_ground(self) -> str:
        """
        ``` yaml
        name: arch_sound_ground
        type: str
        default: vggish
        choices: null
        help: architecture of net_sound_ground
        ```
        """
        return self._arch_sound_ground
    
    @property
    def arch_frame_ground(self) -> str:
        """
        ``` yaml
        name: arch_frame_ground
        type: str
        default: resnet18
        choices: null
        help: architecture of net_frame_ground
        ```
        """
        return self._arch_frame_ground
    
    @property
    def arch_sound(self) -> str:
        """
        ``` yaml
        name: arch_sound
        type: str
        default: unet7
        choices: null
        help: architecture of net_sound
        ```
        """
        return self._arch_sound
    
    @property
    def arch_frame(self) -> str:
        """
        ``` yaml
        name: arch_frame
        type: str
        default: resnet18dilated
        choices: null
        help: architecture of net_frame
        ```
        """
        return self._arch_frame
    
    @property
    def arch_synthesizer(self) -> str:
        """
        ``` yaml
        name: arch_synthesizer
        type: str
        default: linear
        choices: null
        help: architecture of net_synthesizer
        ```
        """
        return self._arch_synthesizer
    
    @property
    def arch_grounding(self) -> str:
        """
        ``` yaml
        name: arch_grounding
        type: str
        default: base
        choices: null
        help: architecture of net_grounding
        ```
        """
        return self._arch_grounding
    
    @property
    def weights_sound_ground(self) -> str:
        """
        ``` yaml
        name: weights_sound_ground
        type: str
        default: ''
        choices: null
        help: weights to finetune net_sound_ground
        ```
        """
        return self._weights_sound_ground
    
    @property
    def weights_frame_ground(self) -> str:
        """
        ``` yaml
        name: weights_frame_ground
        type: str
        default: ''
        choices: null
        help: weights to finetune net_frame_ground
        ```
        """
        return self._weights_frame_ground
    
    @property
    def weights_sound(self) -> str:
        """
        ``` yaml
        name: weights_sound
        type: str
        default: ''
        choices: null
        help: weights to finetune net_sound
        ```
        """
        return self._weights_sound
    
    @property
    def weights_frame(self) -> str:
        """
        ``` yaml
        name: weights_frame
        type: str
        default: ''
        choices: null
        help: weights to finetune net_frame
        ```
        """
        return self._weights_frame
    
    @property
    def weights_synthesizer(self) -> str:
        """
        ``` yaml
        name: weights_synthesizer
        type: str
        default: ''
        choices: null
        help: weights to finetune net_synthesizer
        ```
        """
        return self._weights_synthesizer
    
    @property
    def weights_grounding(self) -> str:
        """
        ``` yaml
        name: weights_grounding
        type: str
        default: ''
        choices: null
        help: weights to finetune net_grounding
        ```
        """
        return self._weights_grounding
    
    @property
    def weights_maskformer(self) -> str:
        """
        ``` yaml
        name: weights_maskformer
        type: str
        default: ''
        choices: null
        help: weights to finetune maskformer
        ```
        """
        return self._weights_maskformer
    
    @property
    def num_channels(self) -> int:
        """
        ``` yaml
        name: num_channels
        type: int
        default: 32
        choices: null
        help: number of channels
        ```
        """
        return self._num_channels
    
    @property
    def num_frames(self) -> int:
        """
        ``` yaml
        name: num_frames
        type: int
        default: 64
        choices: null
        help: number of frames
        ```
        """
        return self._num_frames
    
    @property
    def stride_frames(self) -> int:
        """
        ``` yaml
        name: stride_frames
        type: int
        default: 1
        choices: null
        help: sampling stride of frames
        ```
        """
        return self._stride_frames
    
    @property
    def img_pool(self) -> str:
        """
        ``` yaml
        name: img_pool
        type: str
        default: maxpool
        choices: null
        help: avg or max pool image features
        ```
        """
        return self._img_pool
    
    @property
    def img_activation(self) -> str:
        """
        ``` yaml
        name: img_activation
        type: str
        default: sigmoid
        choices: null
        help: activation on the image features
        ```
        """
        return self._img_activation
    
    @property
    def sound_activation(self) -> str:
        """
        ``` yaml
        name: sound_activation
        type: str
        default: 'no'
        choices: null
        help: activation on the sound features
        ```
        """
        return self._sound_activation
    
    @property
    def output_activation(self) -> str:
        """
        ``` yaml
        name: output_activation
        type: str
        default: sigmoid
        choices: null
        help: activation on the output
        ```
        """
        return self._output_activation
    
    @property
    def binary_mask(self) -> int:
        """
        ``` yaml
        name: binary_mask
        type: int
        default: 0
        choices: null
        help: whether to use bianry masks
        ```
        """
        return self._binary_mask
    
    @property
    def mask_thres(self) -> float:
        """
        ``` yaml
        name: mask_thres
        type: float
        default: 0.5
        choices: null
        help: threshold in the case of binary masks
        ```
        """
        return self._mask_thres
    
    @property
    def loss(self) -> str:
        """
        ``` yaml
        name: loss
        type: str
        default: l1
        choices: null
        help: loss function to use
        ```
        """
        return self._loss
    
    @property
    def weighted_loss(self) -> int:
        """
        ``` yaml
        name: weighted_loss
        type: int
        default: 0
        choices: null
        help: weighted loss
        ```
        """
        return self._weighted_loss
    
    @property
    def log_freq(self) -> int:
        """
        ``` yaml
        name: log_freq
        type: int
        default: 1
        choices: null
        help: log frequency scale
        ```
        """
        return self._log_freq
    
    @property
    def split(self) -> str:
        """
        ``` yaml
        name: split
        type: str
        default: val
        choices: null
        help: val or test
        ```
        """
        return self._split
    
    @property
    def num_gpus(self) -> int:
        """
        ``` yaml
        name: num_gpus
        type: int
        default: 1
        choices: null
        help: number of gpus to use
        ```
        """
        return self._num_gpus
    
    @property
    def batch_size_per_gpu(self) -> int:
        """
        ``` yaml
        name: batch_size_per_gpu
        type: int
        default: 32
        choices: null
        help: input batch size
        ```
        """
        return self._batch_size_per_gpu
    
    @property
    def workers(self) -> int:
        """
        ``` yaml
        name: workers
        type: int
        default: 16
        choices: null
        help: number of data loading workers
        ```
        """
        return self._workers
    
    @property
    def num_val(self) -> int:
        """
        ``` yaml
        name: num_val
        type: int
        default: -1
        choices: null
        help: number of images to evalutate
        ```
        """
        return self._num_val
    
    @property
    def num_vis(self) -> int:
        """
        ``` yaml
        name: num_vis
        type: int
        default: 40
        choices: null
        help: number of images to evalutate
        ```
        """
        return self._num_vis
    
    @property
    def audLen(self) -> int:
        """
        ``` yaml
        name: audLen
        type: int
        default: 65535
        choices: null
        help: sound length for MUSIC
        ```
        """
        return self._audLen
    
    @property
    def audRate(self) -> int:
        """
        ``` yaml
        name: audRate
        type: int
        default: 11025
        choices: null
        help: sound sampling rate
        ```
        """
        return self._audRate
    
    @property
    def stft_frame(self) -> int:
        """
        ``` yaml
        name: stft_frame
        type: int
        default: 1022
        choices: null
        help: stft frame length
        ```
        """
        return self._stft_frame
    
    @property
    def stft_hop(self) -> int:
        """
        ``` yaml
        name: stft_hop
        type: int
        default: 256
        choices: null
        help: stft hop length
        ```
        """
        return self._stft_hop
    
    @property
    def imgSize(self) -> int:
        """
        ``` yaml
        name: imgSize
        type: int
        default: 224
        choices: null
        help: size of input frame
        ```
        """
        return self._imgSize
    
    @property
    def frameRate(self) -> float:
        """
        ``` yaml
        name: frameRate
        type: float
        default: 25
        choices: null
        help: video frame sampling rate
        ```
        """
        return self._frameRate
    
    @property
    def seed(self) -> int:
        """
        ``` yaml
        name: seed
        type: int
        default: 1234
        choices: null
        help: manual seed
        ```
        """
        return self._seed
    
    @property
    def ckpt(self) -> str:
        """
        ``` yaml
        name: ckpt
        type: str
        default: ../data/ckpt
        choices: null
        help: folder to output checkpoints
        ```
        """
        return self._ckpt
    
    @property
    def disp_iter(self) -> int:
        """
        ``` yaml
        name: disp_iter
        type: int
        default: 20
        choices: null
        help: frequency to display
        ```
        """
        return self._disp_iter
    
    @property
    def eval_epoch(self) -> int:
        """
        ``` yaml
        name: eval_epoch
        type: int
        default: 1
        choices: null
        help: frequency to evaluate
        ```
        """
        return self._eval_epoch
    
    @property
    def in_channels(self) -> int:
        """
        ``` yaml
        name: in_channels
        type: int
        default: 32
        choices: null
        help: channels of the input features
        ```
        """
        return self._in_channels
    
    @property
    def MASK_FORMER_HIDDEN_DIM(self) -> int:
        """
        ``` yaml
        name: MASK_FORMER_HIDDEN_DIM
        type: int
        default: 32
        choices: null
        help: Transformer feature dimension
        ```
        """
        return self._MASK_FORMER_HIDDEN_DIM
    
    @property
    def MASK_FORMER_NUM_OBJECT_QUERIES(self) -> int:
        """
        ``` yaml
        name: MASK_FORMER_NUM_OBJECT_QUERIES
        type: int
        default: 20
        choices: null
        help: number of queries
        ```
        """
        return self._MASK_FORMER_NUM_OBJECT_QUERIES
    
    @property
    def MASK_FORMER_NHEADS(self) -> int:
        """
        ``` yaml
        name: MASK_FORMER_NHEADS
        type: int
        default: 8
        choices: null
        help: number of heads
        ```
        """
        return self._MASK_FORMER_NHEADS
    
    @property
    def MASK_FORMER_DROPOUT(self) -> float:
        """
        ``` yaml
        name: MASK_FORMER_DROPOUT
        type: float
        default: 0.1
        choices: null
        help: dropout in Transformer
        ```
        """
        return self._MASK_FORMER_DROPOUT
    
    @property
    def MASK_FORMER_DIM_FEEDFORWARD(self) -> int:
        """
        ``` yaml
        name: MASK_FORMER_DIM_FEEDFORWARD
        type: int
        default: 256
        choices: null
        help: feature dimension in feedforward network
        ```
        """
        return self._MASK_FORMER_DIM_FEEDFORWARD
    
    @property
    def MASK_FORMER_ENC_LAYERS(self) -> int:
        """
        ``` yaml
        name: MASK_FORMER_ENC_LAYERS
        type: int
        default: 0
        choices: null
        help: number of Transformer encoder layers
        ```
        """
        return self._MASK_FORMER_ENC_LAYERS
    
    @property
    def MASK_FORMER_DEC_LAYERS(self) -> int:
        """
        ``` yaml
        name: MASK_FORMER_DEC_LAYERS
        type: int
        default: 6
        choices: null
        help: number of Transformer decoder layers
        ```
        """
        return self._MASK_FORMER_DEC_LAYERS
    
    @property
    def SEM_SEG_HEAD_MASK_DIM(self) -> int:
        """
        ``` yaml
        name: SEM_SEG_HEAD_MASK_DIM
        type: int
        default: 32
        choices: null
        help: mask feature dimension
        ```
        """
        return self._SEM_SEG_HEAD_MASK_DIM
    
    @property
    def mode(self) -> str:
        """
        ``` yaml
        name: mode
        type: str
        default: train
        choices: null
        help: train/eval
        ```
        """
        return self._mode
    
    @property
    def list_train(self) -> str:
        """
        ``` yaml
        name: list_train
        type: str
        default: data/train.csv
        choices: null
        help: null
        ```
        """
        return self._list_train
    
    @property
    def list_val(self) -> str:
        """
        ``` yaml
        name: list_val
        type: str
        default: data/val.csv
        choices: null
        help: null
        ```
        """
        return self._list_val
    
    @property
    def dup_trainset(self) -> int:
        """
        ``` yaml
        name: dup_trainset
        type: int
        default: 40
        choices: null
        help: duplicate so that one epoch has more iters
        ```
        """
        return self._dup_trainset
    
    @property
    def num_epoch(self) -> int:
        """
        ``` yaml
        name: num_epoch
        type: int
        default: 100
        choices: null
        help: epochs to train for
        ```
        """
        return self._num_epoch
    
    @property
    def lr_frame_ground(self) -> float:
        """
        ``` yaml
        name: lr_frame_ground
        type: float
        default: 0.001
        choices: null
        help: LR
        ```
        """
        return self._lr_frame_ground
    
    @property
    def lr_sound_ground(self) -> float:
        """
        ``` yaml
        name: lr_sound_ground
        type: float
        default: 0.001
        choices: null
        help: LR
        ```
        """
        return self._lr_sound_ground
    
    @property
    def lr_frame(self) -> float:
        """
        ``` yaml
        name: lr_frame
        type: float
        default: 0.0001
        choices: null
        help: LR
        ```
        """
        return self._lr_frame
    
    @property
    def lr_sound(self) -> float:
        """
        ``` yaml
        name: lr_sound
        type: float
        default: 0.001
        choices: null
        help: LR
        ```
        """
        return self._lr_sound
    
    @property
    def lr_synthesizer(self) -> float:
        """
        ``` yaml
        name: lr_synthesizer
        type: float
        default: 0.001
        choices: null
        help: LR
        ```
        """
        return self._lr_synthesizer
    
    @property
    def lr_grounding(self) -> float:
        """
        ``` yaml
        name: lr_grounding
        type: float
        default: 0.001
        choices: null
        help: LR
        ```
        """
        return self._lr_grounding
    
    @property
    def lr_maskformer(self) -> float:
        """
        ``` yaml
        name: lr_maskformer
        type: float
        default: 0.0001
        choices: null
        help: LR
        ```
        """
        return self._lr_maskformer
    
    @property
    def lr_steps(self) -> int:
        """
        ``` yaml
        name: lr_steps
        type: int
        default:
        - 40
        - 60
        choices: null
        help: steps to drop LR in epochs
        ```
        """
        return self._lr_steps
    
    @property
    def beta1(self) -> float:
        """
        ``` yaml
        name: beta1
        type: float
        default: 0.9
        choices: null
        help: momentum for sgd, beta1 for adam
        ```
        """
        return self._beta1
    
    @property
    def weight_decay(self) -> float:
        """
        ``` yaml
        name: weight_decay
        type: float
        default: 0.0001
        choices: null
        help: weights regularizer
        ```
        """
        return self._weight_decay
    
    @property
    def weight_decay_maskformer(self) -> float:
        """
        ``` yaml
        name: weight_decay_maskformer
        type: float
        default: 0.0001
        choices: null
        help: weights regularizer
        ```
        """
        return self._weight_decay_maskformer
    
    @property
    def lr_drop_maskformer(self) -> int:
        """
        ``` yaml
        name: lr_drop_maskformer
        type: int
        default: 60
        choices: null
        help: lr_drop
        ```
        """
        return self._lr_drop_maskformer
    
    @property
    def lr_drop_maskformer_1(self) -> int:
        """
        ``` yaml
        name: lr_drop_maskformer_1
        type: int
        default: 60
        choices: null
        help: lr_drop
        ```
        """
        return self._lr_drop_maskformer_1
    
    @property
    def lr_drop_maskformer_2(self) -> int:
        """
        ``` yaml
        name: lr_drop_maskformer_2
        type: int
        default: 80
        choices: null
        help: lr_drop
        ```
        """
        return self._lr_drop_maskformer_2
    
    
    @id.setter
    def id(self, value: str):
        self._id = value
    
    @num_mix.setter
    def num_mix(self, value: int):
        self._num_mix = value
    
    @arch_sound_ground.setter
    def arch_sound_ground(self, value: str):
        self._arch_sound_ground = value
    
    @arch_frame_ground.setter
    def arch_frame_ground(self, value: str):
        self._arch_frame_ground = value
    
    @arch_sound.setter
    def arch_sound(self, value: str):
        self._arch_sound = value
    
    @arch_frame.setter
    def arch_frame(self, value: str):
        self._arch_frame = value
    
    @arch_synthesizer.setter
    def arch_synthesizer(self, value: str):
        self._arch_synthesizer = value
    
    @arch_grounding.setter
    def arch_grounding(self, value: str):
        self._arch_grounding = value
    
    @weights_sound_ground.setter
    def weights_sound_ground(self, value: str):
        self._weights_sound_ground = value
    
    @weights_frame_ground.setter
    def weights_frame_ground(self, value: str):
        self._weights_frame_ground = value
    
    @weights_sound.setter
    def weights_sound(self, value: str):
        self._weights_sound = value
    
    @weights_frame.setter
    def weights_frame(self, value: str):
        self._weights_frame = value
    
    @weights_synthesizer.setter
    def weights_synthesizer(self, value: str):
        self._weights_synthesizer = value
    
    @weights_grounding.setter
    def weights_grounding(self, value: str):
        self._weights_grounding = value
    
    @weights_maskformer.setter
    def weights_maskformer(self, value: str):
        self._weights_maskformer = value
    
    @num_channels.setter
    def num_channels(self, value: int):
        self._num_channels = value
    
    @num_frames.setter
    def num_frames(self, value: int):
        self._num_frames = value
    
    @stride_frames.setter
    def stride_frames(self, value: int):
        self._stride_frames = value
    
    @img_pool.setter
    def img_pool(self, value: str):
        self._img_pool = value
    
    @img_activation.setter
    def img_activation(self, value: str):
        self._img_activation = value
    
    @sound_activation.setter
    def sound_activation(self, value: str):
        self._sound_activation = value
    
    @output_activation.setter
    def output_activation(self, value: str):
        self._output_activation = value
    
    @binary_mask.setter
    def binary_mask(self, value: int):
        self._binary_mask = value
    
    @mask_thres.setter
    def mask_thres(self, value: float):
        self._mask_thres = value
    
    @loss.setter
    def loss(self, value: str):
        self._loss = value
    
    @weighted_loss.setter
    def weighted_loss(self, value: int):
        self._weighted_loss = value
    
    @log_freq.setter
    def log_freq(self, value: int):
        self._log_freq = value
    
    @split.setter
    def split(self, value: str):
        self._split = value
    
    @num_gpus.setter
    def num_gpus(self, value: int):
        self._num_gpus = value
    
    @batch_size_per_gpu.setter
    def batch_size_per_gpu(self, value: int):
        self._batch_size_per_gpu = value
    
    @workers.setter
    def workers(self, value: int):
        self._workers = value
    
    @num_val.setter
    def num_val(self, value: int):
        self._num_val = value
    
    @num_vis.setter
    def num_vis(self, value: int):
        self._num_vis = value
    
    @audLen.setter
    def audLen(self, value: int):
        self._audLen = value
    
    @audRate.setter
    def audRate(self, value: int):
        self._audRate = value
    
    @stft_frame.setter
    def stft_frame(self, value: int):
        self._stft_frame = value
    
    @stft_hop.setter
    def stft_hop(self, value: int):
        self._stft_hop = value
    
    @imgSize.setter
    def imgSize(self, value: int):
        self._imgSize = value
    
    @frameRate.setter
    def frameRate(self, value: float):
        self._frameRate = value
    
    @seed.setter
    def seed(self, value: int):
        self._seed = value
    
    @ckpt.setter
    def ckpt(self, value: str):
        self._ckpt = value
    
    @disp_iter.setter
    def disp_iter(self, value: int):
        self._disp_iter = value
    
    @eval_epoch.setter
    def eval_epoch(self, value: int):
        self._eval_epoch = value
    
    @in_channels.setter
    def in_channels(self, value: int):
        self._in_channels = value
    
    @MASK_FORMER_HIDDEN_DIM.setter
    def MASK_FORMER_HIDDEN_DIM(self, value: int):
        self._MASK_FORMER_HIDDEN_DIM = value
    
    @MASK_FORMER_NUM_OBJECT_QUERIES.setter
    def MASK_FORMER_NUM_OBJECT_QUERIES(self, value: int):
        self._MASK_FORMER_NUM_OBJECT_QUERIES = value
    
    @MASK_FORMER_NHEADS.setter
    def MASK_FORMER_NHEADS(self, value: int):
        self._MASK_FORMER_NHEADS = value
    
    @MASK_FORMER_DROPOUT.setter
    def MASK_FORMER_DROPOUT(self, value: float):
        self._MASK_FORMER_DROPOUT = value
    
    @MASK_FORMER_DIM_FEEDFORWARD.setter
    def MASK_FORMER_DIM_FEEDFORWARD(self, value: int):
        self._MASK_FORMER_DIM_FEEDFORWARD = value
    
    @MASK_FORMER_ENC_LAYERS.setter
    def MASK_FORMER_ENC_LAYERS(self, value: int):
        self._MASK_FORMER_ENC_LAYERS = value
    
    @MASK_FORMER_DEC_LAYERS.setter
    def MASK_FORMER_DEC_LAYERS(self, value: int):
        self._MASK_FORMER_DEC_LAYERS = value
    
    @SEM_SEG_HEAD_MASK_DIM.setter
    def SEM_SEG_HEAD_MASK_DIM(self, value: int):
        self._SEM_SEG_HEAD_MASK_DIM = value
    
    @mode.setter
    def mode(self, value: str):
        self._mode = value
    
    @list_train.setter
    def list_train(self, value: str):
        self._list_train = value
    
    @list_val.setter
    def list_val(self, value: str):
        self._list_val = value
    
    @dup_trainset.setter
    def dup_trainset(self, value: int):
        self._dup_trainset = value
    
    @num_epoch.setter
    def num_epoch(self, value: int):
        self._num_epoch = value
    
    @lr_frame_ground.setter
    def lr_frame_ground(self, value: float):
        self._lr_frame_ground = value
    
    @lr_sound_ground.setter
    def lr_sound_ground(self, value: float):
        self._lr_sound_ground = value
    
    @lr_frame.setter
    def lr_frame(self, value: float):
        self._lr_frame = value
    
    @lr_sound.setter
    def lr_sound(self, value: float):
        self._lr_sound = value
    
    @lr_synthesizer.setter
    def lr_synthesizer(self, value: float):
        self._lr_synthesizer = value
    
    @lr_grounding.setter
    def lr_grounding(self, value: float):
        self._lr_grounding = value
    
    @lr_maskformer.setter
    def lr_maskformer(self, value: float):
        self._lr_maskformer = value
    
    @lr_steps.setter
    def lr_steps(self, value: int):
        self._lr_steps = value
    
    @beta1.setter
    def beta1(self, value: float):
        self._beta1 = value
    
    @weight_decay.setter
    def weight_decay(self, value: float):
        self._weight_decay = value
    
    @weight_decay_maskformer.setter
    def weight_decay_maskformer(self, value: float):
        self._weight_decay_maskformer = value
    
    @lr_drop_maskformer.setter
    def lr_drop_maskformer(self, value: int):
        self._lr_drop_maskformer = value
    
    @lr_drop_maskformer_1.setter
    def lr_drop_maskformer_1(self, value: int):
        self._lr_drop_maskformer_1 = value
    
    @lr_drop_maskformer_2.setter
    def lr_drop_maskformer_2(self, value: int):
        self._lr_drop_maskformer_2 = value
    
