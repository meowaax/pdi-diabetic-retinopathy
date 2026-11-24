# Planejamento de Projeto: Segmentação Multiclasse de Lesões da Retinopatia Diabética

**Uma abordagem híbrida de Processamento Digital de Imagens e Deep Learning**
Projeto final de Processamento Digital de Imagens

- **Proposta central:** investigar se técnicas de pré-processamento digital de imagens melhoram a segmentação de diferentes lesões da retinopatia diabética e se seus efeitos variam de acordo com a categoria segmentada.
- **Abordagem:** pré-processamento tradicional + segmentação com UNet++ (encoder VGG16) + análise complementar de pós-processamento.
- **Dataset principal:** IDRiD, com imagens de fundo de olho e anotações de lesões.
- **Infraestrutura:** complexidade alta, compatível com infraestrutura moderada (Google Colab ou GPU compartilhada).

---

## Resumo executivo

A proposta amplia o estudo de referência, que investigou a segmentação de exsudatos duros e o impacto de data augmentation, para quatro categorias de lesões: microaneurismas, hemorragias, exsudatos duros e exsudatos moles. O núcleo da pesquisa compara representações e técnicas de pré-processamento da imagem de entrada, mantendo a arquitetura de segmentação constante.

Recomenda-se usar o IDRiD como base, uma UNet++ com **encoder VGG16 pré-treinado** como modelo inicial e três configurações de entrada: RGB original, CLAHE e canal verde. A combinação CLAHE + canal verde e o pós-processamento morfológico entram como extensões.

### Pergunta central de pesquisa

O pré-processamento ideal depende da categoria de lesão? Técnicas de contraste e representação de cores beneficiam igualmente microaneurismas, hemorragias, exsudatos duros e exsudatos moles?

### Recomendação de escopo

Núcleo obrigatório: RGB original, CLAHE e canal verde. Combinação de técnicas, pós-processamento e validação externa são extensões condicionadas ao tempo e à disponibilidade dos dados.

---

## 1. Comparação das alternativas

| Alternativa | Descrição | Avaliação |
|---|---|---|
| A: Pré-processamento + multiclasse | Comparar RGB original, CLAHE e canal verde antes da segmentação das quatro classes. | Melhor equilíbrio entre contribuição científica, controle experimental e viabilidade. Recomendada como núcleo. |
| B: Multiclasse + pós-processamento | Aplicar operações morfológicas e análise de componentes às máscaras previstas. | Boa conexão com PDI tradicional; recomendada como experimento complementar. |
| C: Pipeline híbrido completo | Combinar pré-processamento, segmentação multiclasse e pós-processamento. | Abrangente, mas dificulta identificar a contribuição de cada etapa. |

**Decisão:** adotar a Alternativa A como eixo científico e incluir pós-processamento apenas depois de estabelecer os resultados principais.

---

## 2. Proposta de pesquisa

- **Título (PT):** Segmentação Multiclasse de Lesões da Retinopatia Diabética: Uma Abordagem Híbrida de Processamento Digital de Imagens e Deep Learning.
- **Título (EN):** Multi-Class Segmentation of Diabetic Retinopathy Lesions: A Hybrid Digital Image Processing and Deep Learning Approach.

### 2.1 Contextualização e problema

A retinopatia diabética pode provocar alterações nos vasos sanguíneos da retina. A segmentação de lesões em imagens de fundo de olho localiza e delimita regiões associadas à doença. O estudo de referência concentrou-se em exsudatos duros; esta proposta amplia o escopo para quatro categorias, com características visuais e dimensões diferentes.

| Categoria | Características visuais gerais |
|---|---|
| Microaneurismas (MA) | Pequenas estruturas avermelhadas. |
| Hemorragias (HE) | Regiões avermelhadas, geralmente maiores. |
| Exsudatos duros (EX) | Depósitos amarelados e brilhantes. |
| Exsudatos moles (SE) | Áreas esbranquiçadas, frequentemente com bordas menos definidas. |

**Problema de pesquisa:** como diferentes técnicas de pré-processamento e PDI influenciam o desempenho de modelos de Deep Learning na segmentação multiclasse de lesões da retinopatia diabética, considerando as particularidades visuais e o desbalanceamento entre as categorias?

### 2.2 Hipóteses

- **Principal:** a aplicação seletiva de pré-processamento pode melhorar a identificação e delimitação de diferentes lesões, com efeitos distintos conforme as características visuais de cada classe.
- **H1:** técnicas de contraste e iluminação melhoram a segmentação em relação às imagens originais.
- **H2:** os benefícios do pré-processamento variam entre as categorias de lesões.
- **H3:** o pós-processamento morfológico pode reduzir regiões espúrias e melhorar a qualidade das máscaras.

### 2.3 Objetivos

**Objetivo geral:** desenvolver e avaliar uma abordagem híbrida de PDI e Deep Learning para a segmentação multiclasse de lesões associadas à retinopatia diabética em imagens de fundo de olho.

**Objetivos específicos:**

- Investigar técnicas de pré-processamento aplicáveis a imagens de fundo de olho.
- Implementar estratégias de aprimoramento de contraste, iluminação e representação de cores.
- Desenvolver um modelo de segmentação capaz de identificar múltiplas categorias de lesões.
- Avaliar o desempenho individual da segmentação para cada categoria.
- Investigar o impacto do pré-processamento sobre a capacidade de segmentação.
- Aplicar pós-processamento para refinar as máscaras produzidas.
- Comparar resultados por métricas quantitativas e análises visuais.

---

## 3. Datasets e dados

### 3.1 Dataset principal: IDRiD

O Indian Diabetic Retinopathy Image Dataset (IDRiD) é recomendado por ser a base do artigo de referência e por disponibilizar máscaras de segmentação de lesões.

| Característica | Informação |
|---|---|
| Imagens com anotações de segmentação | 81 |
| Divisão oficial | 54 treinamento e 27 teste |
| Resolução original | 4288 × 2848 |
| Máscaras MA | 81 |
| Máscaras HE | 80 |
| Máscaras EX | 81 |
| Máscaras SE | 40 |

A disponibilidade desigual de máscaras exige cuidado: uma máscara ausente não significa ausência da lesão. Cada arquivo deve ser verificado e a perda deve ser mascarada, ignorando a supervisão de uma classe quando sua anotação não existir.

Página oficial: https://idrid.grand-challenge.org/Data/

### 3.2 Datasets secundários (extensões opcionais)

| Dataset | Possível uso | Cuidado |
|---|---|---|
| DDR | Avaliação externa ou expansão dos experimentos. | Verificar compatibilidade entre classes, máscaras e protocolos de anotação. |
| FGADR | Ampliar a quantidade de imagens e investigar generalização. | Conferir versão, anotações disponíveis e correspondência entre categorias. |

**Decisão:** IDRiD como principal. DDR e FGADR ficam como extensões, condicionadas à compatibilidade das anotações e ao tempo disponível.

---

## 4. Metodologia experimental

### Etapa 1: Preparação e análise das imagens

- Carregar imagens RGB e máscaras individuais.
- Verificar a correspondência entre imagens e máscaras e identificar anotações ausentes.
- Analisar a distribuição de pixels por classe e o desbalanceamento.
- Visualizar imagens e regiões anotadas.
- Normalizar as imagens e preparar as máscaras em quatro canais.

**Patches:** considerar patches de 512 × 512 com sobreposição, para preservar estruturas pequenas. A divisão treino/validação/teste ocorre por imagem original, antes da extração, para evitar que recortes da mesma imagem apareçam em conjuntos diferentes.

### Etapa 2: Técnicas de PDI

| Configuração | Operação | Finalidade |
|---|---|---|
| A: RGB original | Imagem original, com normalização. | Baseline de comparação. |
| B: CLAHE | CLAHE no canal de luminância (espaço LAB). | Aprimorar contraste local preservando a informação cromática. |
| C: Canal verde | Extrair o canal verde e replicá-lo nos três canais de entrada. | Representação que pode evidenciar estruturas da retina. |
| D: Combinação (opcional) | Combinar CLAHE e canal verde. | Verificar se as operações têm efeito complementar. |

### Etapa 3: Modelo de segmentação

Utilizar **UNet++ com encoder VGG16 pré-treinado** como configuração inicial. Mantém a conexão com o artigo de referência e evita multiplicar o custo computacional com uma comparação extensa de encoders.

A implementação é multi-label: quatro canais de saída independentes com ativação sigmoid (MA, HE, EX, SE). Isso é compatível com máscaras binárias separadas e não obriga cada pixel a pertencer a uma única classe.

A perda inicial combina Binary Cross-Entropy e Dice Loss, com máscara de validade por classe para ignorar supervisões indisponíveis. É preciso testar que classes sem anotação não contribuem para a perda.

| Parâmetro | Configuração inicial sugerida |
|---|---|
| Arquitetura | UNet++ |
| Encoder | VGG16 pré-treinado |
| Entrada | Patches 512 × 512 |
| Batch size | 4–8, conforme memória disponível |
| Learning rate | Começar em 0,001 ou menor e ajustar na validação |
| Otimizador | Adam |
| Épocas máximas | 50–100 |
| Early stopping | 10 épocas sem melhora |
| Loss | BCE + Dice com supervisão mascarada |

Os valores são pontos de partida. Memória disponível e desempenho na validação orientam os ajustes. O VGG16 é mais pesado que o VGG11 (mais parâmetros e memória), então o batch size pode precisar ficar no limite inferior da faixa.

### Data augmentation (fixa em todos os experimentos)

A combinação de augmentation vem do artigo de referência (CBIC 2025), que testou técnicas isoladas e combinadas para segmentação de exsudatos duros no IDRiD com UNet++. A melhor configuração foi **Horizontal Flip + ElasticTransform + RandomBrightnessContrast** com encoder VGG16 (IoU 0,510; Dice 0,676). Combinações com RGBShift pioraram, e ShiftScaleRotate foi a pior técnica isolada.

- A mesma combinação é aplicada em E1 a E5. A única variável entre E1, E2 e E3 é o pré-processamento.
- A busca por combinações de augmentation não é reproduzida; o resultado do artigo é herdado.
- Ordem fixa: augmentation sobre o RGB original e, depois, o pré-processamento (CLAHE ou canal verde).
- RandomBrightnessContrast e CLAHE atuam sobre contraste e podem se sobrepor. Isso é discutido como limitação.
- O artigo segmenta só EX. Para MA (13 a 2.231 px), é preciso verificar que a ElasticTransform mantém as máscaras alinhadas e não as apaga. Qualquer desvio do artigo fica registrado.
- Parâmetros de treino do artigo: 100 épocas, early stopping com paciência 10, batch 6, learning rate 0,001.

### Etapa 4: Pós-processamento complementar

- Abertura e fechamento morfológico.
- Análise de componentes conexos.
- Remoção de pequenas regiões isoladas, quando justificada.
- Comparação entre máscaras originais e refinadas.

Os parâmetros são definidos no conjunto de validação, nunca ajustados no conjunto de teste.

### Etapa 5: Avaliação

| Métrica | Interpretação |
|---|---|
| Dice | Sobreposição entre predição e referência. |
| IoU | Interseção sobre união. |
| Precisão | Proporção de pixels previstos como lesão que realmente são lesão. |
| Sensibilidade | Proporção de pixels de lesão identificados corretamente. |

O indicador principal é o Dice por classe, além do macro-Dice (peso igual às quatro categorias). Agregar resultados por imagem e apresentar variabilidade (por exemplo, intervalos de confiança), dado o tamanho reduzido do conjunto.

---

## 5. Plano de experimentos

| Exp. | Entrada | Pós-processamento | Objetivo |
|---|---|---|---|
| E1 | RGB original | Não | Estabelecer baseline. |
| E2 | CLAHE | Não | Medir efeito do aprimoramento de contraste. |
| E3 | Canal verde | Não | Medir efeito da representação cromática. |
| E4 (opcional) | CLAHE + canal verde | Não | Avaliar combinação de operações. |
| E5 (complementar) | Melhor configuração anterior | Sim | Avaliar refinamento morfológico. |

**Perguntas respondidas:**

- O pré-processamento melhora a segmentação? Comparar E1, E2 e E3.
- As classes se beneficiam igualmente? Comparar métricas por classe em cada configuração.
- A combinação produz resultado melhor? Comparar E2, E3 e E4, se realizado.
- O pós-processamento melhora as máscaras? Comparar a melhor configuração sem e com pós-processamento.

Manter constantes arquitetura, encoder, divisão dos dados, estratégia de treinamento e critérios de avaliação. Quando viável, repetir os experimentos com mais de uma semente aleatória e relatar a variabilidade.

---

## 6. Ferramentas e viabilidade computacional

| Finalidade | Ferramenta sugerida |
|---|---|
| Linguagem | Python |
| Processamento de imagens | OpenCV |
| Dados e máscaras | NumPy, Pillow |
| Visualização | Matplotlib |
| Deep Learning | PyTorch |
| Arquitetura | Segmentation Models PyTorch |
| Data augmentation | Albumentations |
| Métricas | TorchMetrics ou scikit-learn |
| Ambiente | Google Colab ou GPU compartilhada |

Para controlar o custo: uma única arquitetura principal, um único encoder, patches em vez de imagens completas e mixed precision quando a GPU suportar. Patches aumentam as amostras de treino, mas não substituem a diversidade de imagens independentes.

---

## 7. Limitações e cuidados científicos

- **Quantidade de imagens:** o IDRiD tem poucas imagens anotadas. Patches não equivalem a mais imagens independentes.
- **Desbalanceamento:** as classes ocupam proporções diferentes da imagem; métricas globais podem ocultar desempenho baixo em uma classe.
- **Anotações ausentes:** máscara indisponível não significa ausência da lesão. Verificar por imagem e classe.
- **Generalização:** resultados em uma única base não demonstram desempenho em outros equipamentos ou populações.
- **Vazamento de dados:** separar treino, validação e teste por imagem original antes de extrair patches.
- **Ajuste de parâmetros:** escolher hiperparâmetros e pós-processamento na validação, preservando o teste para a avaliação final.

---

## 8. Contribuição esperada

Identificar se o efeito do pré-processamento depende da categoria de lesão. O projeto não precisa encontrar uma única configuração superior em todas as classes. Um resultado em que CLAHE favoreça certas lesões, o canal verde outras e a imagem original permaneça competitiva em algumas é cientificamente relevante.

A análise pode indicar que uma estratégia universal de pré-processamento não é ideal para segmentação multiclasse e motivar, em trabalhos futuros, pipelines específicos por categoria.
