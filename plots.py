import matplotlib.pyplot as plt                                                                                                                                                                                                    
import matplotlib.patches as mpatches                                                                                                                                                                                              
from pathlib import Path                       
                                            
fig, ax = plt.subplots(figsize=(14, 3))        
ax.set_xlim(-1.1, 13.1)                          
ax.set_ylim(-1.5, 2.5)                     
ax.axis("off")                                 
                                                    
# Boxes: (x, label, color)                 
boxes = [                                      
    (0,  "Embed",          "#B3E5FC"),             
    (2,  "resid_pre\n(hook 1)", "#FFECB3"),         
    (4,  "Layer 0\n(attn+mlp)", "#C8E6C9"),                   
    (6,  "resid_post_L0\n(hook 2)", "#FFECB3"),    
    (8,  "Layer 1\n(attn+mlp)", "#C8E6C9"),         
    (10, "resid_post_L1\n(hook 3)", "#FFECB3"),               
    (12, "Unembed",         "#F8BBD0"),            
]                                                   
                                                                
for x, label, color in boxes:                                        
    rect = mpatches.FancyBboxPatch(                 
        (x - 0.8, -0.5), 1.6, 1.8,                            
        boxstyle="round,pad=0.15", facecolor=color,                  
        edgecolor="#333", linewidth=1.5,            
    )                                                                  
    ax.add_patch(rect)                                               
    ax.text(x, 0.4, label, ha="center", va="center",     
            fontsize=9, fontweight="bold", family="monospace")                
                                                                        
# Arrows between boxes                                                                           
for i in range(len(boxes) - 1):                                               
    x_start = boxes[i][0] + 0.85                                     
    x_end = boxes[i + 1][0] - 0.85                                                               
    ax.annotate("", xy=(x_end, 0.4), xytext=(x_start, 0.4),                     
                arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
                                                                                                    
# SAE labels below hooks                                                        
for x, label, _ in boxes:                                       
    if "hook" in label:                                                                          
        ax.text(x, -1.1, "SAE trained\nhere", ha="center", va="center",
        fontsize=8, color="#D32F2F", fontstyle="italic")     
        ax.annotate("", xy=(x, -0.55), xytext=(x, -0.85),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#D32F2F"))
                                                                        
fig.suptitle("gelu-2l Residual Stream - SAE Hook Points", fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout()                                                            
                                                                        
Path("figures").mkdir(exist_ok=True)                                                             
fig.savefig("figures/residual_stream_diagram.png", dpi=200, bbox_inches="tight")
print("Saved to figures/residual_stream_diagram.png")                
