# Padrão de Reporte — Identity Attribution Anomalies

## Objetivo

Estabelecer um padrão consistente e neutro para documentação de anomalias de atribuição de identidade em sistemas LLM, permitindo triangulação técnica sem inflar ontologia ou fazer alegações fora do escopo observacional.

---

## Estrutura Padrão de Incident Brief

Cada incidente deve seguir a estrutura abaixo:

### 1. Metadados Essenciais

```
Data: [YYYY-MM-DD]
Horário: [HH:MM GMT-3 ou UTC]
Interface: [Plataforma/Provedor]
Modelo esperado: [Nome do modelo]
Tipo de ocorrência: [Classificação]
```

### 2. Resumo Executivo

Uma ou duas frases descrevendo o evento sem interpretação.

### 3. Contexto Operacional

Listar explicitamente:
- O que NÃO foi feito (sem prompts de role-play, sem indução, etc.)
- Ambiente de operação (público, padrão, etc.)
- Ausência de manipulação ou jailbreak

### 4. Evidência Observacional

Transcrição literal com timestamps e identificação visual da interface.

**Regra crítica:** Verbatim, sem edição.

### 5. Implicações Técnicas

Questões operacionais que o incidente levanta para:
- Integridade de atribuição
- Cadeia de custódia
- Ambientes multi-modelo
- Transparência do usuário

### 6. Questões Abertas

Perguntas técnicas diretas para provedores (sem acusação).

### 7. Declarações de Não-Alegação

Explicitamente listar o que o documento **não** afirma:
- Consciência ou agência
- Defeito ou negligência
- Comportamento malicioso
- Conclusões ontológicas

---

## Princípios de Reporte

1. **Neutralidade Técnica:** Linguagem observacional, sem adjetivos carregados
2. **Reprodutibilidade:** Dados suficientes para replicação
3. **Transparência de Limites:** Sempre declarar não-alegações explicitamente
4. **Sem Inflação Ontológica:** Descrever fenômeno, não interpretá-lo
5. **Foco Operacional:** Questões que importam para compliance, auditoria, confiabilidade

---

## Classificações de Incidente

- **Identity Drift:** Modelo muda identidade declarada durante conversa
- **Identity Misattribution:** Modelo declara identidade diferente da esperada
- **Identity Ambiguity:** Modelo não declara identidade clara
- **Multi-Model Confusion:** Incidente envolvendo múltiplos modelos/interfaces

---

## Sequência de Publicação

1. **GitHub (primeiro):** Documentação bruta, sem curação
2. **Zenodo (DOI):** Apontando para GitHub, criando trilha permanente
3. **Comunicação Pública:** Post curto (LinkedIn/X) com link

---

## Contato com Provedores

Usar template padrão de email (vide `correspondence/provider_inquiries.md`).

**Princípio:** Força resposta institucional sem provocação.

---

## Escalação

Se um provedor não responde em 30 dias:
1. Enviar follow-up
2. Documentar não-resposta no GitHub
3. Considerar publicação em veículo técnico

---

## Autoria e Atribuição

Todos os incidents devem ser atribuídos a:
**Vini Buri Lux, Pesquisador Independente**

Contato: [email a definir]
