# Correspondência Institucional — Templates e Histórico

## Email Template — Consulta Técnica Padrão

```
Assunto: Consulta técnica sobre integridade de identidade em LLMs públicos

Prezada equipe de Pesquisa / Trust & Safety,

Estou documentando ocorrências relacionadas à atribuição de identidade declarada 
por modelos LLM em interfaces públicas.

Em uma interação recente com a interface pública da [DeepSeek / Anthropic / OpenAI], 
observei um caso de autoidentificação inconsistente do modelo, descrito no 
Incident Brief #001 anexo.

Minhas perguntas são estritamente técnicas e operacionais:

1. Existem mecanismos que garantem que o modelo se identifique corretamente 
   em todas as respostas?

2. Esses mecanismos operam no nível do modelo ou da interface?

3. Eventos de autoatribuição inconsistente são registrados como anomalia?

4. Como a identidade é mantida em cenários multi-modelo ou com roteamento dinâmico?

O objetivo é compreender boas práticas industriais, não atribuir falha ou 
responsabilidade.

Agradeço desde já qualquer esclarecimento institucional.

Atenciosamente,
Vini Buri Lux
Pesquisador Independente
```

---

## Provedores Alvo (Ordem de Contato)

1. **DeepSeek** (provedor do modelo no incidente)
2. **Anthropic** (Claude, modelo mencionado)
3. **OpenAI** (ChatGPT, contexto comparativo)
4. **Google** (Gemini, ecossistema)
5. **xAI** (Grok, ecossistema)

---

## Histórico de Correspondência

### Incident #001 — DeepSeek

| Data | Provedor | Status | Resposta |
|------|----------|--------|----------|
| [Data] | DeepSeek | Enviado | Pendente |
| [Data] | Anthropic | Enviado | Pendente |
| [Data] | OpenAI | Enviado | Pendente |

---

## Notas sobre Estratégia

- **Não provocar:** Linguagem neutra, sem acusação
- **Forçar resposta:** Questões técnicas específicas que exigem resposta
- **Documentar tudo:** Cada email, resposta e follow-up fica no GitHub
- **Escalar se necessário:** 30 dias sem resposta = follow-up + publicação

---

## Anexos Padrão

Cada email deve incluir:
- Incident Brief completo (PDF ou MD)
- Link para repositório GitHub
- Link para DOI (Zenodo)

---

## Follow-Up Template (30 dias)

```
Assunto: [FOLLOW-UP] Consulta técnica sobre integridade de identidade em LLMs

Prezada equipe,

Há 30 dias enviei uma consulta técnica sobre anomalias de atribuição de 
identidade em LLMs (Incident Brief #001, anexo).

Gostaria de confirmar se a mensagem foi recebida e quando posso esperar 
um retorno.

A documentação está disponível publicamente em:
[GitHub URL]
[Zenodo DOI]

Agradeço a atenção.

Atenciosamente,
Vini Buri Lux
```

---

## Publicação de Não-Resposta

Se um provedor não responder após follow-up:

1. Documentar no GitHub com tag `no-response-[provider]`
2. Publicar post público mencionando:
   - Data de contato
   - Questões técnicas formuladas
   - Falta de resposta
   - Disponibilidade pública da documentação

Exemplo:
```
Contatei [Provedor] em [data] com questões técnicas sobre 
anomalias de atribuição de identidade. Sem resposta até [data].
Documentação pública: [link]
```

---

## Escalação para Mídia

Se houver padrão de não-resposta entre provedores:

1. Preparar press release técnico
2. Contatar jornalistas de tecnologia
3. Publicar em veículos técnicos (ArXiv, Medium, etc.)

Foco: "Ecossistema de LLMs carece de padrões de identidade"
(não: "LLMs estão bugados")
