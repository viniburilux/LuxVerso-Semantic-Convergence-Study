import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

def analyze_timeline(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        timestamps = []
        for conv in data:
            create_time = conv.get('create_time')
            if create_time:
                timestamps.append(datetime.fromtimestamp(create_time))
        
        # Agrupar por mês e ano
        month_year = [t.strftime('%Y-%m') for t in timestamps]
        counts = Counter(month_year)
        
        # Ordenar por data
        sorted_counts = sorted(counts.items())
        months, counts_list = zip(*sorted_counts)
        
        # Plotar gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(months, counts_list, marker='o', linestyle='-', color='orange')
        plt.xticks(rotation=45)
        plt.xlabel('Mês-Ano')
        plt.ylabel('Número de Conversas')
        plt.title('Evolução Temporal das Conversas (20 meses)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Gráfico de evolução temporal salvo em: {output_path}")
        
    except Exception as e:
        print(f"Erro na análise de evolução temporal: {e}")

if __name__ == "__main__":
    analyze_timeline('/home/ubuntu/upload/conversations.json', '/home/ubuntu/human-llm-longitudinal-dataset/results/timeline_analysis_chart.png')
