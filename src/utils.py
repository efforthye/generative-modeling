# 공통 유틸리티 함수 모음
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_neural_network(layer_sizes, layer_names=None, title=None, save_path=None):
    """
    Args:
        layer_sizes: 각 레이어의 노드 수 리스트
        layer_names: 각 레이어 이름 리스트
        title: 그래프 제목 (None이면 save_path에서 추출)
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    ax.set_ylim(-0.5, max(layer_sizes) + 0.5)
    ax.axis('off')
    
    max_nodes_display = 10
    node_positions = []
    
    for layer_idx, layer_size in enumerate(layer_sizes):
        positions = []
        
        if layer_size <= max_nodes_display:
            display_nodes = layer_size
            y_positions = np.linspace(0, max(layer_sizes) - 1, display_nodes)
            y_positions = y_positions - np.mean(y_positions) + max(layer_sizes) / 2
            skip_middle = False
        else:
            display_nodes = max_nodes_display
            y_positions = np.linspace(0, max(layer_sizes) - 1, display_nodes)
            y_positions = y_positions - np.mean(y_positions) + max(layer_sizes) / 2
            skip_middle = True
        
        for i, y in enumerate(y_positions):
            if skip_middle and i == display_nodes // 2:
                ax.text(layer_idx, y, '...', fontsize=16, ha='center', va='center', fontweight='bold')
                positions.append((layer_idx, y, True))
            else:
                circle = plt.Circle((layer_idx, y), 0.15, color='steelblue', ec='black', linewidth=1.5, zorder=10)
                ax.add_patch(circle)
                positions.append((layer_idx, y, False))
        
        node_positions.append(positions)
        
        if layer_names and layer_idx < len(layer_names):
            ax.text(layer_idx, -1.5, f"{layer_names[layer_idx]}\n({layer_size})", 
                   fontsize=10, ha='center', va='top', fontweight='bold')
    
    for layer_idx in range(len(node_positions) - 1):
        current_layer = node_positions[layer_idx]
        next_layer = node_positions[layer_idx + 1]
        
        for (x1, y1, is_dots1) in current_layer:
            if is_dots1:
                continue
            for (x2, y2, is_dots2) in next_layer:
                if is_dots2:
                    continue
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # 제목 설정
    if title is None and save_path:
        title = os.path.splitext(os.path.basename(save_path))[0]
    
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"{save_path} 저장 완료")
    
    plt.show()
    return fig
