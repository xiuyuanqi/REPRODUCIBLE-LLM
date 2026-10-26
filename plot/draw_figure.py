from pathlib import Path
import os,json,math
R=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(R/'mpl_cache');os.environ['XDG_CACHE_HOME']=str(R/'cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap,to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
D=json.loads((R/'plot_data.json').read_text())
FACTOR_DATA=json.loads((R/'panel_d_data.json').read_text())
MODELS=D['models'];SCHEMES=D['schemes'];HW=D['hardware'];B=D['batches']
LABELS=['STD','CoRun','BlockLock-Rows','BlockLock-Packed'];COLORS=['#4679ad','#df9047','#299a81','#875eb0']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8.5,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6.3,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.5,'ytick.major.width':.5})
fig=plt.figure(figsize=(11.6,6.3),facecolor='white')
gs=fig.add_gridspec(2,3,left=.10,right=.99,top=.90,bottom=.145,hspace=.50,wspace=.45,height_ratios=[1.55,1])
axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]
cmap=LinearSegmentedColormap.from_list('agreement',['#cc7643','#f2ddab','#b7d8c6','#167e6b']);cmap.set_bad('#e6e8eb')
lookup={(r['hardware'],r['model'],r['scheme'],r['batch_size']):r for r in D['heatmaps']}
hardware_labels=[]
for ax,metric,title in [(axes[0],'token_agreement','(a) Generated-token agreement'),(axes[1],'top20_agreement','(b) Top-20 logit-trace agreement')]:
 matrix=np.full((len(HW)*5,20),np.nan)
 for h,hw in enumerate(HW):
  for m,model in enumerate(MODELS):
   for s,scheme in enumerate(SCHEMES):
    for b,batch in enumerate(B):
     v=lookup[(hw,model,scheme,batch)][metric]
     if v is not None:matrix[h*5+m,s*5+b]=v
 im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0,vmax=100,interpolation='nearest')
 ax.set_title(title,loc='left',weight='bold',pad=27)
 labels=D['model_labels']*len(HW)
 ax.set_yticks(range(len(labels)),labels,fontsize=6.1);ax.set_xticks(range(20),[str(b) for b in B]*4,fontsize=5.5)
 ax.tick_params(length=0,pad=2.5)
 ax.set_xticks(np.arange(-.5,20,1),minor=True);ax.set_yticks(np.arange(-.5,len(labels),1),minor=True)
 ax.grid(which='minor',color='white',linewidth=.35);ax.tick_params(which='minor',length=0)
 for edge in (4.5,9.5,14.5):ax.axvline(edge,color='white',lw=1.8)
 for edge in (4.5,9.5):ax.axhline(edge,color='white',lw=2)
 for s,label in enumerate(LABELS):ax.text((s*5+2+.5)/20,1.065,label.replace('BlockLock-','BlockLock-\n'),transform=ax.transAxes,ha='center',va='center',fontsize=6.2 if s>1 else 6.6,weight='bold',color=COLORS[s])
 if ax is axes[0]:
  for h,label in enumerate(['A6000\nAmpere','A100\nAmpere','H100\nHopper']):hardware_labels.append(ax.text(-.34,1-(h+.5)/len(HW),label,transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=6.7,weight='bold',color='#53616b'))
 # Match reference: annotate variable STD cells only; all policy colors still use actual data.
 for i in range(len(labels)):
  for j in range(20):
   v=matrix[i,j]
   if j<5 and np.isfinite(v) and v<100-1e-9:
    txt=f'{v:.0f}'
    ax.text(j,i,txt,ha='center',va='center',fontsize=4.8,color='white' if v>85 else '#333b40')
 for i in range(len(labels)):
  for s in range(4):
   if np.isnan(matrix[i,s*5:(s+1)*5]).all():ax.text(s*5+2,i,'n/a',ha='center',va='center',fontsize=5.2,color='#747e87')
 ax.set_xlabel('Batch size B',fontsize=6.2,labelpad=5)
 for sp in ax.spines.values():sp.set_visible(False)
# Color scale geometry measured from Fig4_Time&agreement.pdf (835.2 x 453.6 pt).
# Stored coordinates make the redraw independent of the external reference file.
reference_page_width,reference_page_height=835.2,453.6
colorbar_bounds_pt=(312.211731,230.2942291,369.6538086,237.2542358) # x0, top, x1, bottom
cx0,ctop,cx1,cbottom=colorbar_bounds_pt
cax=fig.add_axes([cx0/reference_page_width,(reference_page_height-cbottom)/reference_page_height,
                  (cx1-cx0)/reference_page_width,(cbottom-ctop)/reference_page_height])
cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[])
cb.ax.set_xticks([]);cb.outline.set_visible(False)
# Endpoint labels are beside the gradient, rather than underneath it.
for label,x0,top,bottom,size in [('Identical run pairs',242.902,231.441,237.441,6),
                                 ('0%',301.726,231.5498,236.8498,5.3),
                                 ('100%',371.972,232.3898,237.6898,5.3)]:
 fig.text(x0/reference_page_width,1-(top+bottom)/(2*reference_page_height),label,
          fontsize=size,ha='left',va='center',color='#53616b')
ax=axes[3];ax.set_title('(d) Recorded prefill + decode time',loc='left',weight='bold',pad=17)
# Broken y-axis: total-runtime view above, same stacked bars enlarged near zero below.
# Both axes retain the original linear values; no prefill multiplier is applied.
d_original_position=ax.get_position().frozen()
ax.set_position([d_original_position.x0,d_original_position.y0+.27*d_original_position.height,
                 d_original_position.width,.73*d_original_position.height])
prefill_ax=fig.add_axes([d_original_position.x0,d_original_position.y0,
                         d_original_position.width,.25*d_original_position.height],sharex=ax)
groups=[('A6000',8),('A100',8),('H100',8)];x=np.arange(3);barw=.185
for s,scheme in enumerate(SCHEMES):
 rows=[next(r for r in D['stacks'] if r['hardware']==hw and r['batch_size']==b and r['scheme']==scheme) for hw,b in groups]
 pre=np.array([r['prefill_ratio'] for r in rows]);dec=np.array([r['decode_ratio'] for r in rows]);xx=x+(s-1.5)*barw
 for phase_ax in (ax,prefill_ax):
  phase_ax.bar(xx,pre,barw,color=COLORS[s],zorder=3)
  phase_ax.bar(xx,dec,barw,bottom=pre,color=to_rgba(COLORS[s],.30),edgecolor=COLORS[s],linewidth=.45,zorder=3)
 if s in (0,1,3):
  for pos,r in zip(xx,rows):ax.text(pos,r['total_ratio']+.10,f"{r['total_ratio']:.2f}",ha='center',va='bottom',fontsize=5.1,color='#43515c')
ax.axhline(1,color='#7e8890',lw=.65,ls='--');ax.set_ylim(.60,7.2);ax.set_yticks([1,2,4,6]);ax.set_ylabel('Recorded CUDA time / STD',fontsize=6.8)
ax.yaxis.set_label_coords(-.10,.26)
prefill_ax.set_xticks(x,[f'{hw}\nQwen3-'+('8B' if hw=='A100' else '14B') for hw,b in groups],fontsize=6.3)
prefill_ax.tick_params(axis='x',length=0)
ax.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
ax.grid(axis='y',color='#e3e8eb',lw=.5,zorder=0)
ax.legend(handles=[Patch(facecolor='#596976',label='Prefill'),Patch(facecolor='#cdd4d9',edgecolor='#596976',label='Decode')],loc='upper right',fontsize=5.8,frameon=False,ncol=2,handlelength=1.2,columnspacing=1)
# The lower view spans only 0--0.022 of the STD total, exposing the solid prefill base.
prefill_ax.set_ylim(0,.022)
prefill_ax.set_yticks([0,.01,.02],['0','0.01','0.02'],fontsize=5.4)
prefill_ax.grid(axis='y',color='#e3e8eb',lw=.5,zorder=0)
ax.spines['bottom'].set_visible(False);prefill_ax.spines['top'].set_visible(False)
# Diagonal marks explicitly identify the omitted interval (0.022, 0.60).
for horizontal_edge in (0,1):
 ax.plot([horizontal_edge-.012,horizontal_edge+.012],[-.012,.012],transform=ax.transAxes,color='#596976',lw=.7,clip_on=False,zorder=8)
 prefill_ax.plot([horizontal_edge-.012,horizontal_edge+.012],[.965,1.035],transform=prefill_ax.transAxes,color='#596976',lw=.7,clip_on=False,zorder=8)
prefill_shares=[100*r['prefill_ratio']/r['total_ratio'] for r in D['stacks']]
assert len(ax.patches)==len(prefill_ax.patches)==24
for main_bar,detail_bar in zip(ax.patches,prefill_ax.patches):
 assert np.allclose([main_bar.get_x(),main_bar.get_y(),main_bar.get_width(),main_bar.get_height()],
                    [detail_bar.get_x(),detail_bar.get_y(),detail_bar.get_width(),detail_bar.get_height()],rtol=0,atol=1e-12)
(R/'panel_d_scale_verification.json').write_text(json.dumps(dict(mode='two linear y ranges with explicit axis breaks',total_view_ylim=[.60,7.2],prefill_view_ylim=[0,.022],omitted_interval=[.022,.60],prefill_percentage_range=[min(prefill_shares),max(prefill_shares)],numeric_values_rescaled=False,stacked_phase_values_identical_in_both_views=True),indent=2))

# CoRun-to-method comparison arrows; factors use unrounded plotted heights.
for group_index,(hw,b) in enumerate(groups):
 ref=next(r['total_ratio'] for r in D['stacks'] if r['hardware']==hw and r['scheme']=='corun_style')
 for scheme,offset,text_offset,color_idx in [('normal',-.39,-.045,0),('scheme1',.115,-.045,2),('scheme2_fixedL_capacity',.39,.045,3)]:
  item=next(v for v in FACTOR_DATA['factors'] if v['hardware']==hw and v['scheme']==scheme)
  low=next(r['total_ratio'] for r in D['stacks'] if r['hardware']==hw and r['scheme']==scheme)
  assert abs(item['CoRun_plotted_height']-ref)<1e-10 and abs(item['method_plotted_height']-low)<1e-10
  xpos=group_index+offset;color=COLORS[color_idx]
  ax.annotate('',xy=(xpos,low),xytext=(xpos,ref),arrowprops=dict(arrowstyle='<->',lw=.65,color=color,shrinkA=0,shrinkB=0),zorder=6)
  ax.plot([xpos-.025,xpos+.025],[ref,ref],color=color,lw=.65,zorder=6)
  ax.text(xpos+text_offset,(ref+low)/2,f"{item['annotated_CoRun_over_method']:.2f}×",rotation=90,ha='center',va='center',fontsize=5.0,color=color,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=.35),zorder=7)

ax=axes[5];ax.set_title('(f) Useful prefill throughput',loc='left',weight='bold',pad=17)
xs=np.arange(5)
for s,scheme in enumerate(SCHEMES):
 for h,hw in enumerate(HW):
  if s==0 and h>0:continue
  yy=[next(r['throughput_ratio'] for r in D['curves'] if r['hardware']==hw and r['batch_size']==b and r['scheme']==scheme) for b in B]
  ax.plot(xs,yy,color=COLORS[s],lw=1.1 if h==0 else 1,ls=['-','-.','--'][h],marker=['o','^','s'][h],ms=2.5 if h==0 else 2.2,zorder=4-s*.1)
ax.set_ylim(0,1.15);ax.set_yticks([0,.25,.5,.75,1]);ax.set_xticks(xs,B);ax.set_xlabel('Batch size B',fontsize=6.8);ax.set_ylabel('Useful input tokens/s / STD',fontsize=6.8);ax.grid(axis='y',color='#e3e8eb',lw=.5,zorder=0)
ax.legend(handles=[Line2D([],[],color='#52616f',marker=mk,ms=2.5,lw=1,ls=ls,label=lab) for mk,ls,lab in zip(['o','^','s'],['-','-.','--'],['A6000 / 14B','A100 / 8B','H100 / 14B'])],loc='lower left',fontsize=5.5,frameon=False,ncol=1,handlelength=2,columnspacing=1,labelspacing=.25)
# New right column: dataset-resolved A6000 results, same model/B/cohort in both panels.
order=D['dataset_detail_order'];xx=np.arange(len(order));dw=.19
ax=axes[2];ax.set_title('(c) A6000 agreement by dataset',loc='left',weight='bold',pad=27)
for s,scheme in enumerate(SCHEMES):
 rr=[next(r for r in D['dataset_detail'] if r['dataset']==ds['dataset'] and r['scheme']==scheme) for ds in order]
 ax.bar(xx+(s-1.5)*dw,[r['token_agreement_pct'] for r in rr],dw,color=COLORS[s],zorder=3)
ax.axhline(100,color='#6c7e87',ls=':',lw=.6,zorder=4)
ax.set_ylim(0,107);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Identical generated-token run pairs (%)',fontsize=6.6)
ax.set_xticks(xx,[ds['label'] for ds in order],rotation=48,ha='right',rotation_mode='anchor',fontsize=6)
ax.tick_params(axis='x',length=0,pad=3);ax.grid(axis='y',color='#e3e8eb',lw=.5,zorder=0)
ax=axes[4];ax.set_title('(e) A6000 decode time by dataset',loc='left',weight='bold',pad=17)
for s,scheme in enumerate(SCHEMES):
 rr=[next(r for r in D['dataset_detail'] if r['dataset']==ds['dataset'] and r['scheme']==scheme) for ds in order]
 ax.plot(xx,[r['decode_time_s'] for r in rr],color=COLORS[s],lw=1.15,ls='--' if s==0 else '-',marker=['o','^','s','D'][s],markerfacecolor='white' if s==0 else COLORS[s],ms=3.4 if s==0 else 2.7,zorder=5 if s==0 else 3)
ax.set_yscale('log');ax.set_ylim(20,250);ax.set_yticks([20,30,50,100,200]);ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:g}'));ax.minorticks_off()
ax.set_ylabel('Decode time',fontsize=6.8)
ax.set_xticks(xx,[ds['label'] for ds in order],rotation=48,ha='right',rotation_mode='anchor',fontsize=6)
ax.tick_params(axis='x',length=0,pad=3);ax.grid(axis='y',which='major',color='#e3e8eb',lw=.5,zorder=0)
fig.legend(handles=[Patch(facecolor=c,label=m) for c,m in zip(COLORS,LABELS)],loc='lower center',bbox_to_anchor=(.55,.025),ncol=4,frameon=False,fontsize=7,columnspacing=1.6,handlelength=1.25)
# Align the left edge of panel (a) 's hardware labels to panel (d) 's y-axis label.
# Indent the heatmap within its slot so model labels remain separate from hardware labels.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
target_left=axes[3].yaxis.label.get_window_extent(renderer).x0
current_left=hardware_labels[0].get_window_extent(renderer).x0
original_position=axes[0].get_position()
shift=(target_left-current_left)/fig.bbox.width/1.34
axes[0].set_position([original_position.x0+shift,original_position.y0,original_position.width-shift,original_position.height])
axes[0].set_title('(a) Generated-token agreement',loc='left',weight='bold',pad=27,x=-shift/(original_position.width-shift))
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
assert all(abs(t.get_window_extent(renderer).x0-target_left)<1.0 for t in hardware_labels)
fig_align=dict(panel_a_hardware_left_px=hardware_labels[0].get_window_extent(renderer).x0,panel_d_ylabel_left_px=target_left,panel_b_hardware_labels=0)
(R/'alignment_verification.json').write_text(json.dumps(fig_align,indent=2))
for ext in ('png','svg','pdf'):fig.savefig(R/f'blocklock_overview_2x3.{ext}',dpi=330 if ext=='png' else 300,facecolor='white')
# Export a standalone detail of panel (d) with identical plotted bars and annotations.
fig.canvas.draw()
panel_d_bbox=Bbox.union([axes[3].get_tightbbox(fig.canvas.get_renderer()),prefill_ax.get_tightbbox(fig.canvas.get_renderer())]).expanded(1.04,1.08).transformed(fig.dpi_scale_trans.inverted())
for ext in ('png','svg','pdf'):fig.savefig(R/f'panel_d_with_factors.{ext}',dpi=330 if ext=='png' else 300,facecolor='white',bbox_inches=panel_d_bbox)
print(R/'blocklock_overview_2x3.png')
