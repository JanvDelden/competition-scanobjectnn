# from .runner import run_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_pretrain import test_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_cls as test_cls