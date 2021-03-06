# Neural Translation Models  

class EmbeddingLM(nn.Module):
    def __init__(self, TEXT, dropout=0.0, max_embedding_norm=None, embedding_size=1000):
        super(EmbeddingLM, self).__init__()
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.vocab_size = len(TEXT.vocab)
        self.embedding_dim = embedding_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, 
                                       max_norm=max_embedding_norm)

class EncoderLSTM(EmbeddingLM):
    def __init__(self, TEXT, hidden_size=500, num_layers=2,
                 bidirectional=False, **kwargs):
        super(EncoderLSTM, self).__init__(TEXT, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout_prob, batch_first=True,
                            bidirectional=self.bidirectional)

        
    def forward(self, input, hidden):
        x = self.embeddings(input)  # batch size, sentence len, embedding dim
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        return output, hidden  # batch, sentence len, hidden size * 1 or 2


class DecoderLSTM(EncoderLSTM):
    """Inherit same architecture as encoder, but reversed"""
    def __init__(self, TEXT, context_size=1, bidirectional_encoder=False, **kwargs):
        super(DecoderLSTM, self).__init__(TEXT, **kwargs)
        self.context_size = context_size
        self.encoder_dirs = 2 if bidirectional_encoder else 1
        input_dim = self.context_size * self.num_layers * self.encoder_dirs + 1
        self.output = nn.Linear(input_dim * self.hidden_size, self.vocab_size)

    def forward(self, input, hidden, context):
        """:context: tuple (h, c) of hidden and cell states from t-1 of encoder"""
        x = self.embeddings(input)
        x = F.relu(x)
        output, hidden = self.lstm(x, hidden)

        if self.context_size:
            context_var = torch.cat(context[:self.context_size])
            batch_size = context_var.size(1)
            sentence_len = output.size(1)

            # Convert this to named version? rn [batch_size, 1, hidden_size * context_size]
            context_var = context_var.permute(1, 0, 2).contiguous().view(batch_size, 1, -1)
            context_var = context_var.expand(-1, sentence_len, -1)
            output = torch.cat((output, context_var), dim=2)

        output = self.output(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden


class DecoderAttn(EncoderLSTM):
    def __init__(self, TEXT, bidirectional_encoder=False, tie_weights=False,
                 linear_encoder=0, **kwargs):
        super(DecoderAttn, self).__init__(TEXT, **kwargs)
        print('Using final MLP')
        self.encoder_dirs = 2 if bidirectional_encoder else 1

        dims = self.encoder_dirs  
        self.output_decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.output_context = nn.Linear(dims * self.hidden_size, self.vocab_size)

        self.linear_encoder = linear_encoder
        if self.linear_encoder > 0:
            self.attn_linear = nn.Linear(self.encoder_dirs * self.linear_encoder,
                                         self.hidden_size)

        if tie_weights:
            if self.hidden_size != self.embedding_dim:
                raise ValueError('Tied weights require hidden size to equal embedding dimension!')
            self.output_decoder.weight = self.embeddings.weight
        
    def forward(self, input, hidden, encoded_output, mask_ix=None):
        embedding = self.embeddings(input)
        embedding = self.dropout(embedding)
        decoder_output, hidden = self.lstm(embedding, hidden)
        
        # Attention
        if self.linear_encoder > 0:
            output_linear_encoder = self.attn_linear(encoded_output)
        else:
            output_linear_encoder = encoded_output

        # output_encoder_prm is [batch_size, hidden_size, sentence_len (src)] <- TODO: Named
        output_encoder_perm = output_linear_encoder.permute(0, 2, 1)
        products = torch.bmm(decoder_output, output_encoder_perm)

        # mask_ix is [batch_size, sentence_len_src]
        if mask_ix is not None:
            mask_ix = Variable(torch.Tensor([np.inf])) * mask_ix
            mask_ix[mask_ix != mask_ix] = 0
            products = products - torch.unsqueeze(mask_ix, 1)
        
        product_softmax = F.softmax(products, dim=2)
        context = torch.bmm(product_softmax, encoded_output)

        output_1 = self.output_decoder(self.dropout(decoder_output))
        output_2 = self.output_context(self.dropout(context))
        
        output = output_1 + output_2
        output = F.log_softmax(output, dim=2)
        
        return output, hidden, product_softmax 