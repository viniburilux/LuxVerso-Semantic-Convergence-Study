import json
import re
from collections import Counter
import matplotlib.pyplot as plt

def analyze_term_frequency(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        user_texts = []
        for conv in data:
            mapping = conv.get('mapping', {})
            for node_id, node in mapping.items():
                message = node.get('message')
                if message and message.get('author', {}).get('role') == 'user':
                    parts = message.get('content', {}).get('parts', [])
                    for part in parts:
                        if isinstance(part, str):
                            user_texts.append(part)
        
        full_text = ' '.join(user_texts).lower()
        words = re.findall(r'\w+', full_text)
        
        # Filtrar stopwords comuns (simplificado)
        stopwords = {'que', 'e', 'de', 'a', 'o', 'eu', 'é', 'não', 'com', 'para', 'do', 'um', 'você', 'uma', 'se', 'da', 'em', 'no', 'pra', 'me', 'mas', 'isso', 'já', 'como', 'agora', 'na', 'gente', 'ele', 'aqui', 'mais', 'por', 'vai', 'tá', 'vou', 'tudo', 'aí', 'só', 'as', 'ou', 'os', 'fazer', 'tem', 'vamos', 'ser', 'ao', 'pelo', 'nos', 'está'}
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        common_words = Counter(filtered_words).most_common(20)
        
        # Plotar gráfico
        words_list, counts = zip(*common_words)
        plt.figure(figsize=(12, 8))
        plt.barh(words_list, counts, color='skyblue')
        plt.xlabel('Frequência')
        plt.title('Top 20 Termos Mais Frequentes (Excluindo Stopwords)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Gráfico de frequência de termos salvo em: {output_path}")
        
    except Exception as e:
        print(f"Erro na análise de frequência de termos: {e}")

if __name__ == "__main__":
    analyze_term_frequency('/home/ubuntu/upload/conversations.json', '/home/ubuntu/human-llm-longitudinal-dataset/results/term_frequency_chart.png')
