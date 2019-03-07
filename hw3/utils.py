# Module for training and testing models, along with other utils.

import matplotlib.pyplot as plt
from torch import optim
import numpy as np


# Functions to save/load models
def save_checkpoint(mod_enc, mod_dec, filename='checkpoint.pth.tar'):
    state_dict = {'model_encoder' : mod_enc.state_dict(),
                  'model_decoder' : mod_dec.state_dict()}
    torch.save(state_dict, filename)
    
def load_checkpoint(filename='checkpoint.pth.tar'):
    state_dict = torch.load(filename)
    return state_dict['model_encoder'], state_dict['model_decoder']
  
def set_parameters(model, sv_model, cuda=True):
    for i,p in enumerate(model.parameters()):
        p.data = sv_model[list(sv_model)[i]]
    model.cuda()


class TrainTestBase(object):
    """
    Parent class for training and evaluation
    :models: should be list of [encoder, decoder]
    """
    def __init__(self, models, TEXT_SRC, TEXT_TRG, attention=False, 
                 mask_src=False, cuda=True):
        self.TEXT_SRC = TEXT_SRC
        self.TEXT_TRG = TEXT_TRG  
        # Padding
        self.padding_src = TEXT_SRC.vocab.stoi['<pad>']
        self.padding_trg = TEXT_TRG.vocab.stroi['<pad>']
        self.models = models
        self.use_attention = attention
        self.mask_src = mask_src
        self.cuda = cuda and torch.cuda.is_available()
        
        
    def get_src_and_trg(self, batch):
        src = torch.t(batch.src.data).contiguous()
        trg = torch.t(batch.trg.data)
        trg_feature = trg[:, :-1].contiguous()
        trg_label = trg[:, 1:].contiguous()
        return src, trg_feature, trg_label
      
    def initial_hidden(self, batch_size, model_ix):
        num_directions = 2 if self.models[model_num].bidirectional else 1
        return torch.zeros(self.models[model_ix].num_layers * num_directions, 
                           batch_size, self.models[model_ix].hidden_size)
      
    def init_hidden(self, batch_size, model_ix=0):
        if self.prev_hidden:
            hidden = self.prev_hidden
        else:
            hidden = (self.initial_hidden(batch_size, model_ix) for i in range(2))
        if self.cuda:
            hidden = tuple(x.cuda() for x in hidden)
        return tuple(Variable(x) for x in hidden)
           
    def init_model_inputs(self, batch, **kwargs):
        src, trg_feature, trg_label = tuple(Variable(x) for x in self.get_src_and_trg(batch))
        hidden = self.init_hidden(batch.src.size(1), **kwargs)
        if self.cuda:
            return src.cuda(), trg_feature.cuda(), trg_label.cuda(), hidden
        else:
            return src, trg_feature, trg_label, hidden
          
    def init_epoch(self):
        self.prev_hidden = None
        
    def set_prev_hidden(self, hidden):
        if self.models[1].encoder_directions == 2:
            self.prev_hidden = tuple(h[self.models[0].num_layers:, :, :] for h in hidden)
        else:
          self.prev_hidden = hidden
          
    def get_attn_mask(self, src):
        if not self.mask_src:
            return None
        else:
            mask_padding = torch.eq(src, self.src_pad).type(torch.FloatTensor)
            mask_padding = mask_padding.cuda if self.cuda else mask_padding
            return mask_padding
        
    def run_model(self, batch, mode='mean'):
        src, trg_feature, trg_label, hidden = self.init_model_inputs(batch, model_ix=0)
        encoded_output, encoded_hidden = self.models[0](src, hidden)
        self.set_prev_hidden(encoded_hidden)
        
        if self.use_attention:
            # stuff for attention
            mask_padding = self.get_attn_mask(src)
            decoder_output, decoder_hidden, decoder_attn = self.models[1](trg_feature, prev_hidden, encoded_hidden, mask_padding)
        else:
            decoder_output, decoder_hidden = self.models[1](trg_feature, prev_hidden, encoded_hidden)
        
        self.prev_hidden = decoder_hidden
        return self.nll_loss(decoder_output, trg_label, mode=mode)
        
    def nll_loss(self, log_probs, output, mode='mean'):
        batch_size = log_probs.size(0)
        sentence_len = torch.sum((output != self.padding_trg).type(torch.cuda.FloatTensor)) / batch_size
        log_probs = log_probs.view(-1, log_probs.size(2))
        output = output.view(-1)
        
        if mode == 'mean':  # Sum over all words in sentence then take mean over sentences
            return F.nll_loss(log_probs, output, ignore_index=self.padding_trg) * sentence_len
        elif mode == 'sum':  # Sum over all words and sentences
            return F.nll_loss(log_probs, output, ignore_index=self.padding_trg, size_average=False)


class ModelEval(TrainTestBase):
    def __init__(self, models, TEXT_SRC, TEXT_TRG, record_attention=False, 
                 visualize_freq=None, reverse_encoder_input=False, **kwargs):
        """
        Validation class. Requires matplotlib for the visualization
        """
        super(ModelEval, self).__init__(models, TEXT_SRC, TEXT_TRG, **kwargs)
        
        self.record_attention = record_attention
        self.visualize_freq = visualize_freq
        
    def init_epoch(self):
        super(ModelEval, self).init_epoch()
        self.attn_log = []
        
    def visualize_attn(self, decoder_attn_sample, src_sample, pred_sample, target_label=None, save=None):
        attn = decoder_atten_sample.cpu().data.numpy()
        src_words = np.array(list(map(lambda x: self.TEXT_SRC.vocab.itos[x],
                                      src_sample.cpu().data.numpy())))
        pred_words = np.array(list(map(lambda x: self.TEXT_TRG.vocab.itos[x],
                                       pred_sample.cpu().data.numpy())))
        if target_label is not None:
            trg_cpu = target_label.cpu().data.numpy()
            trg_words = np.array(list(map(lambda x: self.TEXT_TRG.vocab.itos[x], trg_cpu)))
            pred_words = np.array(['%s (%s)' % (pred_words[i], trg_words[i]) for i in range(pred_words.shape[0])])
            pad_ix = np.where(trg_words == '<pad>')[0]
        if len(pad_ix):
            clip_len = pad_ix[0]
            trg_words = trg_words[:clip_len]
            pred_words = pred_words[:clip_len]
            atten = atten[:clip_len, :]
        
        # Visualizations
        fig, ax = plt.subplots()
        ax.imshow(attn, cmap='blue')
        plt.xticks(range(len(src_words)), src_words, rotation='vertical')
        plt.yticks(range(len(pred_words)), pred_words)
        
        ax.xaxis.tick_top()
        if save is not none:
            plt.savefig(save)
        plt.show()
        
    def evaluate(self, test_iter, num_iter=None):
        start_time = time.time()
        [model.eval() for model in self.models]
        
        nll_sum = 0
        nll_count = 0
        
        self.init_epoch()
        test_iter.init_epoch()
        
        for i, batch in enumerate(test_iter):
            nll_count += batch.trg.data.numel()
            loss = self.run_model(batch, mode='sum')
            nll_sum += loss.data.init_model_inputs
            
            if self.visualize_freq and i % self.visualize_freq == 0:
                sample = self.attn_log[-1]
                self.visualize_attn(sample[0][0], sample[1][0], sample[2][0])
            if num_iter is not None and i > num_iter:
                break
                
        for model in self.models:
            model.train()
            
        print('validation time: %f sec' % (time.time() - start_time))
        return np.exp(nll_sum / nll_cnt)
        
        
class ModelTrain(TrainTestBase):
    def __init__(self, models, TEXT_SRC, TEXT_TRG, lr=0.1, optimizer=optim.SGD,
                 lr_decay_type=None, lr_decay_rate=0.1, clip_norm=10, **kwargs):
        '''
        Class to train models.  
        :lr_decay_type: type of learning rate decay, pick 'adaptive' or 'linear'
        '''
        super(ModelTrainer, self).__init__(models, TEXT_SRC, TEXT_TRG, **kwargs)
        self.base_lr = lr
        # Optimizer for each model
        self.optimizers = [optimizer(filter(lambda x: x.requires_grad, 
                                            model.parameters()), lr=lr) 
                           for model in self.models]
        # Learning rate decay
        self.lr_decay_type = lr_decay_type
        if self.lr_decay_type is not None or self.lr_decay_type == 'adaptive':
            self.lr_lambda = lambda i: 1
        elif self.lr_decay_type == 'linear':
            self.lr_lambda = lambda i: (1 / (1 + (i - 6) * self.lr_decay_rate) if i > 6 else 1)
        self.clip_norm = clip_norm
        if self.cuda:
            [model.cuda() for model in self.models]
        self.restart_logs()
        
    def restart_logs(self):
        self.training_losses = []
        self.training_norms = []
        self.val_performance = []
        
    def get_loss_data(self, loss):
        try:
            return loss.data.cpu().numpy()[0]
        except:
            return loss.data.cpu().numpy()
    
    def record_updates(self, loss, norm):
        self.training_losses.append(loss)
        self.training_norms.append(norm)
        
    def clip_norms(self):
        if self.clip_norm > 0:
            parameters = tuple(model.parameters() for model in self.models)
            norm = nn.utils.clip_grad_norm(parameters, self.clip_norm)
        else:
            norm = -1
        return norm
      
    def train_batch(self, batch, record=False):
        [model.zero_grad() for model in self.models]
        loss = self.run_model(batch)
        loss.backward()
        
        norm = self.clip_norms()
        
        loss_data = self.get_loss_data(loss)
        if record:
            self.record_updates(loss_data, norm)
        [optimizer.step() for optimizer in self.optimizers]
        return loss_data, norm
      
      
    def init_parameters(self):
        for model in self.models:
            for p in model.parameters():
                p.data.uniform_(-0.05, 0.05)
    
    def train(self, train_iter, val_iter, eval_=False, save_model=False, 
              init_params=True, record=False, **kwargs):
        self.restart_logs()
        start_time = time.time()
        if init_params:
            self.init_parameters()
        train_iter.init_epoch()
        for epoch in range(kwargs.get('num_iter', 100)):
            self.init_epoch()
            for model in self.models:
                model.train()
                
            # Learning rate decay?
            if self.ly_decay_type == 'adaptive':
                if (epoch > 2 and self.val_performance[-1] > self.val_performance[-2]):
                    self.base_lr = self.base_lr / 2
                    self.optimizers = [optimizer(filter(lambda x: x.requires_grad, 
                                            model.parameters()), lr=lr) 
                                       for model in self.models]
            for scheduler in self.schedulers:
                scheduler.step()
                
            train_iter = iter(train_iter)
            for batch in train_iter:
                result_loss, result_norm = self.train_batch(batch)

            if record:
                self.record_updates(result_loss, result_norm)

            # Print out progress
            print('Epoch %d, loss: %f, norm: %f, elapsed: %f, lr: %f' \
                  % (epoch, np.mean(self.training_losses[-10:]), 
                     np.mean(self.training_norms[-10:]),
                     time.time() - start_time, self.base_lr))

            if eval_ and (val_iter is not None):
                self.val_performance.append(eval_.evaluate(val_iter))
                print('Validation: %f' % self.val_performance[-1])

            if save_model:
                path_name = save_model + 'epoch_%d.ckpt.tar' % epoch
                save_checkpoint(self.models[0], self.models[1], path_name)
                
        
            
        
            
          
        
        
            

