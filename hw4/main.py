# PyTorch Implementation of A Decomposable Attention Model
import os
import time
import argparse
from tqdm import tqdm
import traceback

import torch
import torchtext
from torch.nn.utils import clip_grad_norm_
from namedtensor import ntorch

from models.attention import AttentionModel, NamedAttentionModel
from models.mixture import LatentMixtureModel, VAE
from models.setup import train_iter, val_iter, test, TEXT


def get_args():
    parser = argparse.ArgumentParser(
        description='Decomposable Attention Model')
    parser.add_argument(
        '--algo', default='attention'
    )
    parser.add_argument(
        '--epochs', type=int, default=100
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0
    )
    parser.add_argument(
        '--grad_clip', type=float, default=5.
    )
    parser.add_argument(
        '--elbo', default='reinforce'
    )
    parser.add_argument(
        '--log_freq', type=int, default=10000
    )
    parser.add_argument(
        '--save_file', default='./trained_models/attn-0.pt'
    )
    parser.add_argument(
        '--intra_attn', default='false'
    )
    parser.add_argument(
        '--pred_suffix', default=''
    )
    parser.add_argument(
        '--visualize_freq', type=int, default=10000
    )
    parser.add_argument(
        '--save_img', default='attn-0'
    )
    # assert args.algo in ['attention', 'ensemble', 'vae']
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args


args = get_args()

device = torch.device("cuda:0" if args.cuda else "cpu")


def get_correct(preds, labels):
    return (labels == preds).sum().item()


# Code from last pset and modified
def visualize_attn(self, src_sample, pred_sample,
                   trg_label=None, save_name=None):
    # attn = decoder_attn_sample.cpu().data.numpy()
    src_words = np.array(list(map(lambda x: TEXT.vocab.itos[x],
                                  src_sample.cpu().data.numpy())))
    pred_words = np.array(list(map(lambda x: TEXT.vocab.itos[x],
                                   pred_sample.cpu().data.numpy())))

    # Visualizations
    fig, ax = plt.subplots()
    ax.imshow(attn, cmap='blue')
    plt.xticks(range(len(src_words)), src_words, rotation='vertical')
    plt.yticks(range(len(pred_words)), pred_words)

    ax.xaxis.tick_top()

    if not save_name is None:
        plt.savefig(save_name)
    # plt.show()


def evaluate(model, batches):
    model.eval()
    with torch.no_grad():
        loss_fn = ntorch.nn.NLLLoss(reduction='sum').spec('label')
        total_loss = 0
        total_num = 0
        num_correct = 0
        for i, batch in enumerate(batches):
            log_probs, attn_a, attn_b = model.forward(
                batch.premise, batch.hypothesis)
            preds = log_probs.argmax('label')
            total_loss += loss_fn(log_probs, batch.label).item()
            num_correct += get_correct(preds, batch.label)
            total_num += len(batch)

            if args.visualize_freq and i % self.visualize_freq == 0:
                fname = './img/' + args.save_img
                visualize_attn(attn_a, attn_b, fname)

        return total_loss / total_num, 100.0 * num_correct / total_num


def train(model, num_epochs, lr, weight_decay, grad_clip,
          log_freq, save_file):
    if os.path.exists(save_file):
        model.load_state_dict(torch.load(save_file))

    val_loss, val_acc = evaluate(model, val_iter)
    opt = torch.optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    loss_fn = ntorch.nn.NLLLoss().spec('label')

    # Save best performing models
    best_params = {k: v.detach().clone() for k, v in model.named_parameters()}
    best_val_acc = val_acc

    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0
        total_num = 0
        num_correct = 0

        state_dict = opt.state_dict()
        for params in state_dict['param_groups']:
            params['lr'] = args.lr
        opt.load_state_dict(state_dict)

        try:  # Actually train
            model.train()
            for i, batch in enumerate(tqdm(train_iter), 1):
                epoch_start = time.time()
                opt.zero_grad()
                log_probs, attn_a, attn_b = model.forward(
                    batch.premise, batch.hypothesis)
                preds = log_probs.detach().argmax('label')
                loss = loss_fn(log_probs, batch.label)

                total_loss += loss.detach().item()
                total_num += len(batch)
                num_correct += get_correct(preds, batch.label)

                loss.backward()
                clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

                # Logging
                if i % args.log_freq == 0:
                    epoch_end = time.time()
                    print(
                        'Epoch {} | Batch Progress: {:.4f} | Training Loss: {:.4f} | Training Acc: {:.4f} | Time: {:.4f}'
                        .format(epoch, i / len(train_iter), total_loss / log_freq, num_correct / total_num,
                                float(epoch_end - epoch_start)))
                    total_loss = 0
                    total_num = 0
                    num_correct = 0

            # Validation
            model.eval()
            val_loss, val_acc = evaluate(model, val_iter)
            if val_acc > best_val_acc:
                best_params = {k: v.detach().clone()
                               for k, v in model.named_parameters()}
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_file)

        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                print(f'Got unexpected interrupt: {e!r}')
                traceback.print_exc()

            print(f'\nStopped training after {epoch} epochs...')
            break

    model.load_state_dict(best_params)
    print('Val Loss: {:.4f} | Val Acc: {:.4f} | Time: {:.4f}'.format(
        val_loss, val_acc, time.time() - start_time))


def train_vae(model, num_epochs, lr=1e-3, weight_decay, grad_clip,
              log_freq, save_file):
    """Function to train VAEs. Could probably refactor with enough overlap to the above"""

    if os.path.exists(save_file):
        model.load_state_dict(torch.load(save_file))

    val_loss, val_acc = evaluate(model, val_iter)

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # Save best performing models
    best_params = {k: v.detach().clone() for k, v in model.named_parameters()}
    best_val_acc = val_acc

    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0

        try:  # Actually train
            model.train()
            for i, batch in enumerate(tqdm(train_iter), 1):
                epoch_start = time.time()
                opt.zero_grad()

                loss, elbo = model.get_loss(
                    batch.premise, batch.hypothesis, batch.label)

                try:
                    total_loss += elbo.detach().item()
                except:
                    total_loss += elbo.item()

                loss.backward()
                clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

                # Logging
                if i % args.log_freq == 0:
                    epoch_end = time.time()
                    print(
                        'Epoch {} | Batch Progress: {:.4f} | Training Loss: {:.4f} | Time: {:.4f}'
                        .format(epoch, i / len(train_iter), total_loss / log_freq,
                                float(epoch_end - epoch_start)))
                    total_loss = 0

            # Validation
            model.eval()
            val_loss, val_acc = evaluate(model, val_iter)
            if val_acc > best_val_acc:
                best_params = {k: v.detach().clone()
                               for k, v in model.named_parameters()}
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_file)

        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                print('Unexpected interrupt: {}'.format(e))
                traceback.print_exc()

            print(f'\nStopped training after {epoch} epochs...')
            break

    model.load_state_dict(best_params)
    print('Val Loss: {:.4f} | Val Acc: {:.4f} | Time: {:.4f}'.format(
        val_loss, val_acc, time.time() - start_time))

def visualize_attn_1(attn_model, sent_1, sent_2, labels=None, save_name=None):
    preds, attn_1, attn_2 = attn_model.forward(sent_1, sent_2, labels)
    sent_1_words = np.array(list(lambda x: TEXT.vocab.itos[x], sent_1.cpu().data.numpy()))
    sent_2_words = np.array(list(lambda x: TEXT.vocab.itos[x], sent_2.cpu().data.numpy()))

    fig, ax = plt.subplots()
    ax.imshow(attn_1, cmap='gray')
    plt.xticks(range(len(sent_1_words)), sent_1_words, rotation='vertical')
    plt.yticks(range(len(sent_2_words)), sent_2_words)

    ax.xaxis.tick_top()

    if not save_name is None:
        plt.savefig(save_name)
    plt.show()

def visualize_attn_2(attn_model, sent_1, sent_2, labels=None, save_name=None):
    preds, attn_1, attn_2 = attn_model.forward(sent_1, sent_2, labels)
    sent_1_words = np.array(list(lambda x: TEXT.vocab.itos[x], sent_1.cpu().data.numpy()))
    sent_2_words = np.array(list(lambda x: TEXT.vocab.itos[x], sent_2.cpu().data.numpy()))

    fig, ax = plt.subplots()
    ax.imshow(attn_2, cmap='gray')
    plt.xticks(range(len(sent_2_words)), sent_2_words, rotation='vertical')
    plt.yticks(range(len(sent_1_words)), sent_1_words)

    ax.xaxis.tick_top()

    if not save_name is None:
        plt.savefig(save_name)
    plt.show()



def get_predictions(model):
    test_iter = torchtext.data.BucketIterator(
        test, train=False, batch_size=10, device=device)

    preds = []
    num_correct = 0
    total_num = 0

    with torch.no_grad():
        model.eval()
        for batch in test_iter:
            batch_preds = model(
                batch.premise, batch.hypothesis).argmax('label')
            preds += batch_preds.tolist()
            num_correct += get_correct(batch_preds, batch.label)
            total_num += len(batch_preds)

    print('Test Acc: {:1f}%'.format(100. * num_correct / total_num))

    with open('predictions.txt', 'w') as f:
        f.write('Id,Category\n')
        for i, pred in enumerate(preds):
            f.write('{},{}\n'.format(str(i), str(pred)))


if __name__ == '__main__':
    if args.algo == 'attention':
        attn = True if args.intra_attn == 'true' else False
        model = NamedAttentionModel(
            num_layers=2, hidden_size=200, dropout=0.2, intra_attn=attn)
        model.cuda()

        train(model, num_epochs=args.epochs, lr=args.lr,
              weight_decay=args.weight_decay, grad_clip=args.grad_clip,
              log_freq=args.log_freq, save_file=args.save_file)

    elif args.algo == 'ensemble':
        # Experiment with different setups - default might be 2 intra, 2 reg attn
        m1 = NamedAttentionModel(
            num_layers=2, hidden_size=200, dropout=0.2, intra_attn=True)
        m2 = NamedAttentionModel(
            num_layers=2, hidden_size=200, dropout=0.2, intra_attn=True)
        m3 = NamedAttentionModel(
            num_layers=2, hidden_size=200, dropout=0.2, intra_attn=False)
        m4 = NamedAttentionModel(
            num_layers=2, hidden_size=200, dropout=0.2, intra_attn=False)

        model = LatentMixtureModel(m1, m2, m3, m4, **kwargs)
        model.cuda()
        train(model, num_epochs=args.epochs, lr=args.lr,
              weight_decay=args.weight_decay, grad_clip=args.grad_clip,
              log_freq=args.log_freq, save_file=args.save_file)

    else:  # assume VAE training
        num_models = 4
        models = [
            NamedAttentionModel(num_layers=2, hidden_size=200,
                                dropout=0.2, intra_attn=False)
            for i in range(num_models)
        ]
        q = NamedAttentionModel(
            num_layers=2, hidden_size=200, dropout=0.2, intra_attn=False, labels=True)
        model = VAE(q, *models, num_samples=1,
                    kl_weight=0.33, elbo_method=args.elbo)

        train_vae(  # wd = 0, gc = 20
            model, lr=1e-3, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            log_freq=args.log_freq, save_file=args.save_file, num_epochs=args.epochs)

    get_predictions(model)


# Reference script command
# python main.py --algo 'vae' --pred_suffix '_vae-0' --grad_clip 20. --save_file './trained_models/vae-0.pt'
