"""Compare observed optimization outcomes under a matching training protocol."""
from . import store

def compare(baseline_id,variant_id,target_reduction=.3,tolerance=.01):
    a,b=store.get('job',baseline_id),store.get('job',variant_id)
    if a['id']==b['id']:raise ValueError('Choose two different runs')
    if any(j['kind']!='train' or j['status']!='completed' for j in (a,b)):raise ValueError('Both runs must be completed training tasks')
    mismatches=[]
    for key in ('dataset_id','data_pipeline','seed','epochs'):
        if a['payload'].get(key)!=b['payload'].get(key):mismatches.append(key)
    for key in ('device','device_name'):
        if a.get(key)!=b.get(key):mismatches.append(key)
    for key in ('torch','cuda'):
        if a.get('environment',{}).get(key)!=b.get('environment',{}).get(key):mismatches.append('environment.'+key)
    def effective(j):return j.get('effective_batch_size',j['payload'].get('batch_size',32))*j.get('effective_accumulation',j['payload'].get('gradient_accumulation',1))
    if effective(a)!=effective(b):mismatches.append('effective optimizer batch')
    if len(a.get('metrics',[]))!=len(b.get('metrics',[])):mismatches.append('completed epoch count')
    peaks=[j.get('memory_peak_mb') for j in (a,b)];available=all(v and v>0 for v in peaks) and str(a.get('device')).startswith('cuda')
    reduction=1-peaks[1]/peaks[0] if available else None
    scores=[j.get('test_accuracy') for j in (a,b)];delta=scores[1]-scores[0] if all(v is not None for v in scores) else None
    matched=not mismatches
    return dict(baseline=baseline_id,variant=variant_id,comparable=matched,mismatches=mismatches,device=a.get('device'),baseline_peak_mb=peaks[0],variant_peak_mb=peaks[1],memory_reduction=reduction,target_reduction=target_reduction,memory_target_observed=matched and available and reduction>=target_reduction,baseline_test_accuracy=scores[0],variant_test_accuracy=scores[1],accuracy_delta=delta,allowed_accuracy_drop=tolerance,quality_preservation_observed=matched and delta is not None and delta>=-tolerance,precision=[j.get('amp_effective',False) for j in (a,b)],micro_batches=[j.get('effective_batch_size') for j in (a,b)],effective_batch=effective(a),evidence='Actual CUDA allocator peaks and held-out accuracy from these runs. Observed outcomes under one recorded protocol; not a guarantee for other data or seeds.')
