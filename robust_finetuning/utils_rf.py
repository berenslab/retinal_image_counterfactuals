from datetime import datetime
import torch
import math


from torch import nn
from robustness.tools.custom_modules import SequentialWithArgs

def ft(model_name, model_ft, num_classes, additional_hidden=0):

    if model_name in ["resnet", "resnet18", "resnet50", "wide_resnet50_2", "wide_resnet50_4", "resnext50_32x4d", 'shufflenet']:
        num_ftrs = model_ft.fc.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        ##inplanes = 64
        ##nchannels = 1
        ##model_ft.conv1 = nn.Conv2d(nchannels, inplanes, kernel_size=7, stride=2, padding=3,
        ##                       bias=False)

        if additional_hidden == 0:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.fc = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, num_classes)
            )
        input_size = 224
    elif model_name == "alexnet":
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif "vgg" in model_name:
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet":
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name in ["mnasnet", "mobilenet"]:
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name in ['convnext']:
        num_ftrs = model_ft.head.fc.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        ##inplanes = 64
        ##nchannels = 1
        ##model_ft.conv1 = nn.Conv2d(nchannels, inplanes, kernel_size=7, stride=2, padding=3,
        ##                       bias=False)

        if additional_hidden == 0:
            model_ft.head.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.head.fc = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, num_classes)
            )
    elif model_name in ['vit-s']:
        num_ftrs = model_ft.head.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        ##inplanes = 64
        ##nchannels = 1
        ##model_ft.conv1 = nn.Conv2d(nchannels, inplanes, kernel_size=7, stride=2, padding=3,
        ##                       bias=False)

        if additional_hidden == 0:
            model_ft.head = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.head = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, num_classes)
            )

    else:
        raise ValueError("Invalid model type, exiting...")

    return model_ft

def get_runname(args):
    args.fname = 'model_{}'.format(str(datetime.now()))
    args.fname += ' {} lr={:.5f} {} ep={}{} attack={} fts={} seed={} RATIO_lam={}'.format(
        args.dataset, #+ ' ' if args.dataset != 'cifar10' else ''
        args.lr_max, args.lr_schedule, args.epochs, ' wd={}'.format(
            args.weight_decay) if args.weight_decay != 5e-4 else '',
        args.attack, #' act={}'.format(args.topcl_act) if not args.finetune_model or args.fts_idx == 'rand' else ''
        args.model_name if args.finetune_model else 'rand', #args.fts_idx
        args.seed,
        args.lambda_ratio)
    args.fname += ' at={}'.format(args.l_norms)
    #args.l_norms = args.l_norms.split(' ')
    if not args.l_eps is None:
        args.fname += ' eps={}'.format(args.l_eps)
    else:
        args.fname += ' eps=default'
    args.fname += ' iter={}'.format(args.at_iter if args.l_iters is None else args.l_iters)


def stats_dict(args):
    stats = {#'rob_acc_test': torch.zeros([args.epochs]),
        #'clean_acc_test': torch.zeros([args.epochs]),
        #'rob_acc_train': torch.zeros([args.epochs]),
        #'loss_train': torch.zeros([args.epochs]),
        'rob_acc_test_dets': {},
        'rob_acc_train_dets': {},
        'loss_train_dets': {},
        'freq_in_at': {},
        }
    #
    for norm in args.all_norms + ['union', 'clean']:
        stats['rob_acc_test_dets'][norm] = torch.zeros([args.epochs])
        stats['rob_acc_train_dets'][norm] = torch.zeros([args.epochs])
        if not norm in ['union']:
            stats['loss_train_dets'][norm] = torch.zeros([args.epochs])
        if not norm in ['union', 'clean']:
            stats['freq_in_at'][norm] = torch.zeros([args.epochs])
    return stats


def load_pretrained_models(modelname):
    from model_zoo.fast_models import PreActResNet18
    model = PreActResNet18(10, activation='softplus1').cuda()
    ckpt = torch.load('./models/{}.pth'.format(modelname))
    model.load_state_dict(ckpt)
    model.eval()
    return model

def get_model(args, ds):
    '''Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to
    fit the target dataset) and a checkpoint.The checkpoint is set to None if noe resuming training.
    '''
    finetuned_model_path = os.path.join(
        args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    if args.resume and os.path.isfile(finetuned_model_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path)
    else:
        if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
            model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args.arch](
                    args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
                dataset=datasets.ImageNet(''), resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys())
            checkpoint = None
        else:
            model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=ds,
                                                          resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained)
            checkpoint = None

        if not args.no_replace_last_layer and not args.eval_only:
            print(f'[Replacing the last layer with {args.additional_hidden} '
                  f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, ds.num_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
        else:
            print('[NOT replacing the last layer]')
    return model, checkpoint


def get_eps_schedule(args, eps_dict):
    assert len(args.l_norms) == 1, 'Eps schedule is only for 1 norm currently.'

    norm_name = args.l_norms[0]
    eps_base = eps_dict[args.dataset][norm_name] / 3.

    def eps_schedule(t):
        if t <= 10:
            return eps_base
        elif 10 < t <= 20:
            return eps_base*2
        else:
            return eps_base*3

    return eps_schedule


def get_lr_schedule(args):
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
        # lr_schedule = lambda t: np.interp([t], [0, args.epochs], [0, args.lr_max])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'piecewise-ft':
        def lr_schedule(t):
            if t / args.epochs < 1. / 3.:
                return args.lr_max
            elif t / args.epochs < 2. / 3.:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule.startswith('piecewise'):
        w = [float(c) for c in args.lr_schedule.split('-')[1:]]
        def lr_schedule(t):
            c = 0
            while t / args.epochs > sum(w[:c + 1]) / sum(w):
                c += 1
            return args.lr_max / 10. ** c

    return lr_schedule


def get_accuracy_and_logits(model, x, y, batch_size=100, n_classes=10):
    logits = torch.zeros([y.shape[0], n_classes], device='cpu')
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].cuda()
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].cuda()

            output = model(x_curr)
            logits[counter * batch_size:(counter + 1) * batch_size] += output.cpu()
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0], logits


