import torch
import math
from typing import Callable, Tuple
import time
def bipartite_soft_matching_ToMe_atten(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    if_prune: bool = False,
    if_order: bool = False,
    shuffle_rate: float = 0.0,
    distance: str = 'cosine'
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    
    def do_nothing(x, mode=None) -> Tuple[torch.Tensor, int]:
        return x
    
    if r <= 0:
        return do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        if distance == 'cosine':
            scores = a @ b.transpose(-1, -2)
        elif distance == 'l1':
            scores = -torch.cdist(a, b, p=1)
        elif distance == 'l2':
            scores = -torch.cdist(a, b, p=2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
      
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        if not if_prune:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def orderdmerge(x: torch.Tensor, mode="mean") -> torch.Tensor:
    
        n, t, c = x.shape
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        idx_origin = torch.arange(t, device=x.device).unsqueeze(0).expand(n, t).unsqueeze(-1)
    
        src_idx_original, dst_idx_original = idx_origin[..., ::2, :], idx_origin[..., 1::2, :]
        
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
   
        src_idx_original = src_idx_original.gather(dim=-2, index=unm_idx)


        if if_prune is False:

            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        original_idx = torch.cat([src_idx_original, dst_idx_original], dim=1)

        sorted_idx, idx = original_idx.sort(dim=1)        
        seq = torch.cat([unm, dst], dim=1) 
        
 
        seq = seq.gather(dim=-2, index=idx.expand(n,seq.shape[1],c))
        if shuffle_rate > 0:

            total_tokens = seq.shape[1]
            num_shuffle = int(total_tokens * shuffle_rate)
            shuffle_indices = torch.randperm(total_tokens)[:num_shuffle]
            
       
            shuffled_seq = seq[:, shuffle_indices]
            shuffled_seq = shuffled_seq[:, torch.randperm(shuffle_indices.shape[0])]
            seq[:, shuffle_indices] = shuffled_seq               
        
        return seq

    if if_order:
        return orderdmerge
    else:
        return merge




def bipartite_soft_matching_ToMe(
    metric: torch.Tensor,
    class_token: bool = False,
    distill_token: bool = False,
    token_position: int = 0,
    num_prune: int = 5,
    if_prune: bool = False,
    if_order: bool = True,
    shuffle_rate: float = 0.0,
    distance: str = 'cosine'
) -> Callable:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    r = num_prune
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)#divide into 2 parts

    def do_nothing(x, mode=None) -> Tuple[torch.Tensor, int]:
        return x, token_position

    if r <= 0:
        return do_nothing

    metric = torch.cat([metric[..., :token_position, :], metric[..., token_position+1:, :]], dim=1)
    
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]#split the tensor into 2 parts
        if distance == 'cosine':
            scores = a @ b.transpose(-1, -2)#calculate the scores
        elif distance == 'l1':
            scores = -torch.cdist(a, b, p=1)
        elif distance == 'l2':
            scores = -torch.cdist(a, b, p=2)
   
        
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  
        src_idx = edge_idx[..., :r, :] 
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) 
        

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        cls_token = x[:,token_position,:]
        x = torch.cat([x[..., :token_position, :], x[..., token_position+1:, :]], dim=1)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        if not if_prune:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

      
        new_token_position = int(token_position * (t - r) // t )
        unm = torch.cat([unm[:,:new_token_position],cls_token.unsqueeze(1),unm[:,new_token_position:]],dim=1)
        return torch.cat([unm, dst], dim=1),new_token_position

    def orderdmerge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        
        n, t, c = x.shape
        cls_token = x[:,token_position,:]
        x = torch.cat([x[..., :token_position, :], x[..., token_position+1:, :]], dim=1)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        idx_origin = torch.arange(t-1, device=x.device).unsqueeze(0).expand(n, t-1).unsqueeze(-1)
        src_idx_original, dst_idx_original = idx_origin[..., ::2, :], idx_origin[..., 1::2, :]
        
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))


        src_idx_original = src_idx_original.gather(dim=-2, index=unm_idx)

    
        if if_prune is False:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            
        
        original_idx = torch.cat([src_idx_original, dst_idx_original], dim=1)

        sorted_idx, idx = original_idx.sort(dim=1)        
        seq = torch.cat([unm, dst], dim=1) 
        

        seq = seq.gather(dim=-2, index=idx.expand(n,seq.shape[1],c))

        new_token_position = int(token_position * (t - r) // t )

        seq = torch.cat([seq[:,:new_token_position],cls_token.unsqueeze(1),seq[:,new_token_position:]],dim=1)
        
        if shuffle_rate > 0:

            total_tokens = seq.shape[1]
            num_shuffle = int(total_tokens * shuffle_rate)
            shuffle_indices = torch.randperm(total_tokens)[:num_shuffle]
            

            shuffled_seq = seq[:, shuffle_indices]
            shuffled_seq = shuffled_seq[:, torch.randperm(shuffle_indices.shape[0])]
            seq[:, shuffle_indices] = shuffled_seq        
        
        return seq,new_token_position

    if if_order:
        return orderdmerge
    else:
        return merge 

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x,token_position = merge(x * size, mode="sum")
    size,_ = merge(size, mode="sum")

    x = x / size
    return x, size, token_position


def merge_wavg_vit(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x =merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source,_ = merge(source, mode="amax")
    return source


    
