def evaluate(model, gen, im, device, image_id=None, show_all_layers=False, show_raw_attn=False):

    # 평균-분산 정규화 (train dataset의 통계량을 (test) input image에 사용
    img=transform(im).unsqueeze(0).to(device)
    
    # model 통과

    outputs =model(img)
    
    # 정확도 70% 이상의 예측만 사용
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] # background 제외
    keep = probas.max(-1).values > 0.7
    
    if keep.nonzero().shape[0] <=1 : # detect된 object
        return
    
    #* 한 개는 evalute 할 필요가 없나요
    
    # 원래 cuda에 적재되어있던 좌표들
    outputs['pred_boxes'] = outputs['pred_boxes'].cpu()
    
    # [0,1]의 상대 좌표를 원래의 좌표로 복구
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)


    #attention weight 저장

    conv_features_in, enc_attn_in, dec_attn_in = [], [], []
    conv_features_out, enc_attn_out, dec_attn_out = [], [], []

    # 이 때, output[0] : [token, 1, hidden_dim] --> 토큰
    # output[1] : [1, token, token] --> 가중치
    hooks = [
#         # real bacbkone (backbone[-1] : positional embeddings)
#         model.backbone[-2].register_forward_hook(
#         lambda self, input, output: conv_features.append(output)
#         ),
        
        #transformer encoder 내 마지막 layer의 self attention 층
#         model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
#         lambda self, input, output: enc_attn_weights.append(output)
#                 ),
        
#         # transformer decoder 내 마지막 layer의 multihead_attn 층
#         model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
#         lambda self, input, output: dec_attn_weights.append(output)
#         ),
        
#         model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
#         lambda self, input, output: test_weights.append(output)
#         )
    
    ]
       
    ## Cnv
    # in
    for layer_name in model.backbone[-2].body:
        hook=model.backbone[-2].body[layer_name].register_forward_hook(
        lambda self, input, output : conv_features_in.append(input)
        )
        hooks.append(hook)
    
    hook=model.backbone[-1].register_forward_hook(
    lambda self, input, output : conv_features_in.append(input))
    #out
    for layer_name in model.backbone[-2].body:
        hook=model.backbone[-2].body[layer_name].register_forward_hook(
        lambda self, input, output : conv_features_out.append(output)
        )
        hooks.append(hook)
    
    hook=model.backbone[-1].register_forward_hook(
    lambda self, input, output : conv_features_out.append(output))
            
    # transformer encoder 내 모든 layer의 output 저장
    # default : (enc_layer = 6)
    
    ## encoder
    # in
    for layer in model.transformer.encoder.layers:
        hook=layer.self_attn.register_forward_hook(
        lambda self, input, output : enc_attn_in.append(input)
        )
        hooks.append(hook)
        
        
        
    # out    
    for layer in model.transformer.encoder.layers:
        hook=layer.self_attn.register_forward_hook(
        lambda self, input, output : enc_attn_out.append(output)
        )
        hooks.append(hook)
        
        
    ## decoder
    # in
    for layer in model.transformer.decoder.layers:
        hook=layer.self_attn.register_forward_hook(
        lambda self, input, output : dec_attn_in.append(input)
        )
        hooks.append(hook)
    # out
    for layer in model.transformer.decoder.layers:
        hook=layer.self_attn.register_forward_hook(
        lambda self, input, output : dec_attn_out.append(output)
        )
        hooks.append(hook)
    # 모델 통과(및 저장)
    model(img)
    
#     return enc_attn_weights
    # hook 제거
    for hook in hooks:
        hook.remove()
    
    # 리스트는 필요 없다.
    # 우선 위의 for문을 통해 encoder layer의 모든 가중치를 저장은 해놨으나,
    # 여기서는 encoder의 마지막 layer만을 사용하자.
#     conv_features = conv_features[0] # feature-map
#     enc_attn_weights = enc_attn_weights[-1] # 마지막 encoder layer의 가중치만 사용 (256개의 값)
#     dec_attn_weights = dec_attn_weights[0] # 마지막 decoder layer의 가중치만 사용 (256개의 값)
    
    #  get the shape of feature map
#     return conv_features
    h, w = conv_features_out[-1].shape[-2:] # Nested tensor -> tensors
# #     img_np = np.array(im).astype(np.float)
    if not show_all_layers == True:
        fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22,7))
    else:
        n_layers=len(model.transformer.encoder.layers)
        if not show_raw_attn:
            fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=n_layers+1, figsize=(22, 4*n_layers))
        else:
            fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=model.transformer.nhead+1,
                                    figsize=(22, 4*model.transformer.nhead))
    # object queries는 100차원(default)이기 때문에 그 중에 
    # 0.7(default) 이상의 신뢰도를 보이는 query만을 사용해야 한다. 
    
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        
        ax = ax_i[0]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin.detach(), ymin.detach()), 
                                  xmax.detach() - xmin.detach(),
                                   ymax.detach() - ymin.detach(), 
                                   fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
        
      
        
        if not show_all_layers == True:
            ax = ax_i[1]
                            
            
            cam = gen.generate_ours(img, idx, use_lrp=False)
            cam = (cam - cam.min()) / (cam.max() - cam.min()) # 점수 정규화
            cmap = plt.cm.get_cmap('Blues').reversed()

            ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap)
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
        else:
            
            if not show_raw_attn:    
                cams = gen.generate_ours(img, idx, use_lrp=False, use_all_layers=True)
            else:
                cams = gen.generate_raw_attn(img, idx, use_all_layers=True)
            
            num_layer=n_layers
            if show_raw_attn:
                num_layer=model.transformer.nhead
            for n, cam in zip(range(num_layer), cams):
                ax = ax_i[1+n]
                cam = (cam - cam.min()) / (cam.max() - cam.min()) # 점수 정규화
                cmap = plt.cm.get_cmap('Blues').reversed()

                ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap)
                ax.axis('off')
                ax.set_title(f'query id: {idx.item()}, layer:{n}', size=12)
        

#     id_str = '' if image_id == None else image_id
#     fig.tight_layout()
#     plt.show


####### ExplanationGenerator.py

class Generator:
    def __init__(self, model):
        print('### Generator.init()')
        self.model = model
        self.model.eval() # evaluate 시 dropout, batchnorm 등은 사용하지 않는다.
        self.use_all_layers=False
        self.use_all_layers=False
        

    def forward(self, input_ids, attention_mask):
#         print('### Generator.forward(input_ids, attention_mask)')
        return self.model(input_ids, attention_mask)

    def generate_transformer_att(self, img, target_index, index=None):
        outputs = self.model(img)
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot.clone().detach()
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs['pred_logits'])

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(one_hot_vector, **kwargs)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        # initialize relevancy matrices
        image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
        queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]

        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)

        # R_q_i generated from last layer
        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn_cam().detach()
        grad_q_i = decoder_last.multihead_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.self_attn.get_attn_gradients().detach() # [8 x wh x wh]의 gradient(당연히 가중치랑 같음)
            if self.use_lrp: # 타당성 전파 시
                cam = blk.self_attn.get_attn_cam().detach() # [8 x wh x wh]의 cam(*)
            else:
                cam = blk.self_attn.get_attn().detach()
            cam = avg_heads(cam, grad)  # [wh x wh]의 averaged cam
            
            
            self.R_i_i += torch.matmul(cam, self.R_i_i)# [wh x wh] X [wh x wh]
            
            if self.use_all_layers == True:
                self.R_i_i_all.append(self.R_i_i.detach().clone())
                
    def handle_co_attn_self_query(self, block):
        grad = block.self_attn.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.self_attn.get_attn_cam().detach()
        else:
            cam = block.self_attn.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam) # 식 (6)(7), 행렬곱
        self.R_q_q += R_q_q_add
        self.R_q_i += R_q_i_add
        
        if self.use_all_layers == True:
            self.R_q_q_all.append(self.R_q_q.detach().clone())

    def handle_co_attn_query(self, block):
        if self.use_lrp:
            cam_q_i = block.multihead_attn.get_attn_cam().detach()  # multihead
        else:
            cam_q_i = block.multihead_attn.get_attn().detach()
        grad_q_i = block.multihead_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i) # = [100 x wh]
        self.R_q_i += apply_mm_attention_rules(self.R_q_q, self.R_i_i, cam_q_i, # 식 (10), 행렬곱(R_ii x cam xi R_q_q)
                                               apply_normalization=self.normalize_self_attention, # R_qq, R_ii 정규화
                                               apply_self_in_rule_10=self.apply_self_in_rule_10) # R_sq 대신 cam_sq (*)
        if self.use_all_layers == True:
            self.R_q_i_all.append(self.R_q_i.detach().clone())
    def generate_ours(self, img, target_index, index=None, use_lrp=True,
                     normalize_self_attention=True, apply_self_in_rule_10=True, use_all_layers=False):
        

        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10
        self.use_all_layers = use_all_layers

        outputs = self.model(img)
        outputs = outputs['pred_logits']
        

        kwargs = {"alpha": 1, 
                 "target_index": target_index}
        
        if index == None:
            index = outputs[0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index
        
        one_hot = torch.zeros_like(outputs).to(outputs.device)
        one_hot[0, target_index, index] = 1 # [1, 100, 92] 차원으로 된 원핫벡터
        one_hot_vector = one_hot # 나중에 타당성 전파하기 위함

        one_hot.requires_grad_(True) # 그래디언트 추적 # 복사 후 원본은 다시 autograd가 추적해야 한다.
        one_hot = torch.sum(one_hot.cuda() * outputs)
        
       
        
        self.model.zero_grad() # 모델 내 그래디언트를 0으로 초기화

        one_hot.backward(retain_graph=True) # backward를 하는 동안에 중간 가중치들은 보존한다.
        
        
        if use_lrp:
            return 

            self.model.relprop(one_hot_vector, **kwargs)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        # 픽셀 개수
        image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1] # wh in (8, wh, wh)
        # object 개수
        queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1] # 100 in (8,100,100)
        # 타당성 행렬(Relevancy matrices) 초기화
        # 또한, 계산 자체가 attention weights랑 행해지기 때문에 device 맞춰주기
        # image self attention matrix - (wh x wh) 크기의 Identity 행렬
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
        
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
        # image --> queries의 영향 ( (100 x wh) 크기 행렬)
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
        
        if self.use_all_layers == True:
            self.R_i_i_all = []
            self.R_q_q_all = []
            self.R_q_i_all = []
        
         # encoder 내에서 image에 대한 self-attention

        self.handle_self_attention_image(encoder_blocks)
    
        # decoder 내에서 queries에 대한 self-attention + Multi-modal attention

        for idx, blk in enumerate(decoder_blocks):
            # decoder self attention
            self.handle_co_attn_self_query(blk)

            # encoder decoder attention
            self.handle_co_attn_query(blk)
      
        
        if not self.use_all_layers:
            aggregated = self.R_q_i.unsqueeze_(0)
            aggregated = aggregated[:,target_index, :].unsqueeze_(0).detach()
            # 결과적으로 위 타겟에 대한 [1, 1, 1, wh] 크기의 cam(if not use_all_layers)
            # 굳이 이렇게 만드는 이유는 아마 [layer, target_index, ...] 를 위해?            
        else:
            # [6, 1, 100, wh]

            aggregated = torch.stack(self.R_q_i_all).unsqueeze_(1)
            # [6, 1, 1, 1, wh]
            aggregated = aggregated[:, :, target_index, :].unsqueeze_(1).detach()

        
        
        return aggregated 

    
    def generate_raw_attn(self, img, target_index, use_all_layers=False):
        outputs = self.model(img)
        
        
        
        # get cross attn cam from last decoder layer
        cam_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        #         
        if use_all_layers:
            self.R_q_i_all = cam_q_i
            aggregated = self.R_q_i_all.unsqueeze_(1)
            aggregated = aggregated[:, :,  target_index, :].unsqueeze_(1)
        else : 
            cam_q_i = cam_q_i.mean(dim=0)
            self.R_q_i = cam_q_i
            aggregated = self.R_q_i.unsqueeze_(0)

            aggregated = aggregated[:, target_index, :].unsqueeze_(0)


        return aggregated
   