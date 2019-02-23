def batch_to_input(batch, n=3):
    data = batch.text.view(-1).data.tolist()
    context = [tuple(data[ix: ix + n]) for ix in range(0, len(data) - n + 1)]
    x = Variable(torch.LongTensor([c[:-1] for c in context]))
    y = Variable(torch.LongTensor([c[-1] for c in context]))
    return x.cuda(), y.cuda()
  
def string_to_batch(string):
    rel_words = string.split()
    ids = [word_to_id(word) for word in rel_words]
    return Variable(torch.LongTensor(ids)).cuda()
  
def word_to_id(word, TEXT=TEXT):
    return TEXT.vocab.stoi[word]
    
def train_model(model, train_iter, optimizer, criterion, lr=1e-3):  
    """
    Train the actual model
    :optimizer: try out torch.optim.Adam  
    :criterion: nn.NLLLoss
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optimizer(params = parameters, lr=lr)
    
    model.train()
    epoch_loss = []
    for batch in train_iter:
        x, y = batch_to_input(batch)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    return epoch_loss


def train_model_lstm(model, hidden, train_iter, optimizer, criterion, lr=1e-3, clip=2):  
    """
    Train the actual model for LSTMs (requires hidden initiation)
    :optimizer: try out torch.optim.Adam  
    :criterion: nn.CrossEntropyLoss
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optimizer(params=parameters, lr=lr)
    
    model.train()
    epoch_loss = []
    for batch in train_iter:
        x, y = batch_to_input(batch)
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_pred, hidden = model.forward(x, hidden, train=True)
        loss = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.lstm.parameters(), clip)
        optimizer.step()
        epoch_loss.append(loss.item())
    return epoch_loss

            
def validate_model(model, val_iter, criterion):
    """
    Evaluate model
    :criterion: nn.NLLLoss()
    """
    model.eval()
    val_loss = []
    for batch in val_iter:
        x, y = batch_to_input(batch)
        probs = model(x)
        val_loss.append(criterion(probs, y).item())
    return val_loss

def train_val_model(model, n_epochs, optimizer, lr=1e-3, lstm=False, batch_size=10):
    """Wrapper to call training and evaluation"""
    criterion = nn.NLLLoss()
    for epoch in tqdm_notebook(range(n_epochs)):  # loady bois  
        if lstm:
            hidden = model.init_hidden(batch_size)
        epoch_loss = train_model(model, train_iter, optimizer, criterion, lr)
        val_loss = validate_model(model, val_iter, criterion)
        # Perplexity
        train_ppl = np.exp(np.mean(epoch_loss))
        val_ppl = np.exp(np.mean(val_loss))
        
        print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(
            epoch + 1, np.mean(epoch_loss), train_ppl,  val_ppl))
                
        
def predict_sentence(model, sentence):
    context = sentence[:-4]
    x = string_to_batch(context)
    return model.predict(x)
        

def write_to_kaggle(model, input_file='input.txt'):
    inputs = open(input_file, 'r').read().splitlines()
    predictions = [predict_sentence(model, sentence) for sentence in inputs]
    with open('predictions_nnlm_01.txt', 'w') as f:
        print('id, word', file=f)
        f.write('id, word')
        for ix, line in enumerate(outputs):
            print('%d, %s'%(i, " ".join(predictions)), file=f)
        
