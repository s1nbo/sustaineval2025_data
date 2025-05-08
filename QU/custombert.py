class CustomBert(torch.nn.Module):
    '''Custom Bert Modell, 
        - Wörterzahl als Feature aufnehmen
        - labels und superlabels verarbeiten
        - context und target getrennt aufnehmen 
        - target mit dynamisch lernbarem Skalierungsfaktor, startet bei Startgewichtung von 1.2 
    '''
    def __init__(self, num_labels, num_superclasses, additional_feature_dim=0, pretrained = 'bert-base-german-cased', target_scaling_start = 1.2):
        super(CustomBert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.dropout = torch.nn.Dropout(0.1)

        input_dim = self.bert.config.hidden_size + additional_feature_dim
        self.fc_features = torch.nn.Linear(input_dim, 512)
        self.fc_label = torch.nn.Linear(512, num_labels)
        self.fc_super = torch.nn.Linear(512, num_superclasses)

        # Dynamisch lernbarer Skalierungsfaktor für Target (Segment 1)
        self.segment1_scaling = torch.nn.Parameter(torch.tensor(target_scaling_start))  # Startet bei 1.2

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, additional_features=None, labels=None, super_labels=None, **kwargs):
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)

        # Segment 1 skalieren
        token_type_embeddings = token_type_embeddings.clone()
        token_type_embeddings[token_type_ids == 1] *= self.segment1_scaling

        inputs_embeds = (
            self.bert.embeddings.word_embeddings(input_ids) +
            self.bert.embeddings.position_embeddings(torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)) +
            token_type_embeddings)

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask)

        pooled_output = outputs.pooler_output

        if additional_features is not None:
            combined = torch.cat((pooled_output, additional_features), dim=1)
        else:
            combined = pooled_output

        x = self.dropout(self.fc_features(combined))

        logits_label = self.fc_label(x)
        logits_super = self.fc_super(x)

        loss = None
        if labels is not None and super_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits_label, labels) + 0.5 * loss_fct(logits_super, super_labels)

        return SequenceClassifierOutput(loss=loss, logits=logits_label)