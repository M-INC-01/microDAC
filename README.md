## microDAC — White Paper Tecnico

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Mattimax-brightgreen)](https://huggingface.co/Mattimax)
[![M.INC](https://img.shields.io/badge/M.INC-Official-blue)](https://huggingface.co/MINC01)

Autore: [Mattia / Mattimax](https://huggingface.co/Mattimax)  
Sito aziendale: [M.INC](https://sites.google.com/view/mattimax-site/home-page)

---
![microDAC Logo](https://raw.githubusercontent.com/M-INC-01/microDAC/main/microDAC_Logo.png)
---

1. Introduzione

microDAC è un modello linguistico autoregressivo progettato da M.INC. per la generazione di risposte conversazionali in lingua italiana.  
Ottimizzato per deployment su hardware consumer e edge, microDAC bilancia efficienza computazionale, qualità linguistica e modularità architetturale.  
Il modello è stato addestrato su un corpus supervisionato di dialoghi e istruzioni, con masking selettivo per focalizzare l’apprendimento sulle risposte dell’assistente.

---

2. Architettura del Modello

microDAC-40M è un modello decoder-only ispirato alla famiglia GPT, con le seguenti caratteristiche:

| Componente              | Valore                        |
|-------------------------|-------------------------------|
| Tipo                    | Transformer decoder-only      |
| Parametri totali        | ~70 milioni                   |
| Vocab size              | 32.000                        |
| Context window          | 2048 token                    |
| Embedding dimension     | 512                           |
| Numero layer            | 16                            |
| Attention heads         | 8                             |
| Feedforward dimension   | 2048                          |
| Attivazione             | GELU (variante gelu_new)    |
| Positional encoding     | Learned (GPT-style)           |
| Precisione              | FP16/BF16                     |
| Gradient checkpointing  | Attivo                        |
| Causal masking          | Applicato                     |
| Loss function           | Causal LM loss (token-level)  |

Note architetturali

- Efficienza: la scelta di 512 dimensioni embedding e 16 layer consente un buon trade-off tra capacità espressiva e footprint computazionale.
- Modularità: la struttura è compatibile con LoRA, quantizzazione post-training e pruning strutturale.
- Scalabilità: il modello può essere esteso fino a 120M parametri mantenendo la compatibilità con tokenizer e dataset.

---

3. Tokenizer

Il tokenizer è un Byte-Pair Encoding (BPE) addestrato da zero su dati italiani.  
È progettato per segmentare efficacemente testo colloquiale, tecnico e istruzionale.

Specifiche

| Parametro               | Valore                        |
|-------------------------|-------------------------------|
| Tipo                    | BPE                           |
| Vocab size              | 32.000                        |
| Pre-tokenizer           | ByteLevel (con spazio iniziale) |
| Post-processor          | ByteLevel (offset trimming)   |
| Token ignoto            | <|unk|>                     |
| Token speciali          | <|user|>, <|assistant|>, <|sep|>, <|pad|>, <|bos|>, <|eos|> |

Funzione dei marker

- <|user|>: inizio turno utente  
- <|assistant|>: inizio risposta modello  
- <|sep|>: fine turno  
- <|pad|>: padding per batch uniformi  
- <|bos|>, <|eos|>: delimitatori opzionali per generazione

---

4. Dataset e Preprocessing

Il dataset è composto da turni di dialogo supervisionati, provenienti da fonti conversazionali e istruzionali.  
Ogni esempio è strutturato come:

`json
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
`

Preprocessing

- Formattazione: ogni turno è convertito in stringa con marker speciali:
  - <|user|> ... <|sep|>  
  - <|assistant|> ... <|sep|>
- Tokenizzazione: truncation a 2048 token, no padding in fase di preprocessing.
- Split: 99% training, 1% validazione (minimo 1000 esempi).

---

5. Strategia di Addestramento

Obiettivo

Il modello è addestrato per generare risposte coerenti e contestuali, evitando la ripetizione del prompt utente.  
La loss è calcolata solo sui token dell’assistente, mascherando quelli dell’utente.

Masking selettivo

- I token tra <|user|> e <|sep|> sono mascherati con -100 nei label.
- I token tra <|assistant|> e <|sep|> sono ottimizzati con causal LM loss.

Collator

- Padding dinamico per batch uniformi.
- Masking applicato a livello di token.
- Conversione in tensori PyTorch.

---

6. Parametri di Training

| Parametro               | Valore                        |
|-------------------------|-------------------------------|
| Epochs                  | 2                             |
| Batch per device        | 8                             |
| Gradient accumulation   | 4                             |
| Learning rate           | 3e-4                          |
| Weight decay            | 0.01                          |
| Scheduler               | Cosine                        |
| Warmup ratio            | 0.05                          |
| Max grad norm           | 1.0                           |
| Precisione              | FP16/BF16                     |
| Optimizer               | AdamW (torch)                 |
| Gradient checkpointing  | Attivo                        |
| Group by length         | Attivo                        |
| Resume da checkpoint    | Automatico                    |
| Save best model         | Attivo                        |

---

7. Fine-tuning e Estensioni

microDAC è progettato per essere facilmente adattabile tramite:

🔧 Fine-tuning

- LoRA / PEFT: compatibile con adapter su attention e feedforward.
- Quantizzazione: supporta 8-bit e 4-bit post-training.
- Pruning: possibile pruning strutturale su layer e heads.

🧪 Estensioni

- Multi-turno: concatenazione di turni <|user|>...<|assistant|>... con masking solo sull’ultimo.
- Contesto esteso: supporto fino a 2048 token, estendibile a 4096 con modifiche minime.
- Integrazione vocale: compatibile con pipeline TTS/STT per agenti vocali.

---

8. Considerazioni tecniche

- Il modello è stabile su GPU consumer (≥12 GB VRAM) con gradient checkpointing.
- Il tokenizer è compatibile con Hugging Face Transformers e può essere riutilizzato per modelli più grandi.
- La struttura è compatibile con FlashAttention e ottimizzazioni CUDA.

---

9. Conclusione

microDAC rappresenta una soluzione compatta, efficiente e specializzata per la generazione conversazionale in italiano.  
La sua architettura è pensata per essere estendibile, adattabile e facilmente integrabile in pipeline industriali.  
Ogni componente — dal tokenizer alla loss — è progettato per massimizzare la qualità linguistica e la robustezza tecnica.

Per estensioni, deployment o integrazione in sistemi M.INC., consultare il team AI interno o il repository tecnico associato.
