"""Gera notebooks/colab_runner.ipynb: guia de execução dos treinos no Google Colab."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Treinos no Google Colab\n\n"
       "Guia de execução da sensibilidade do CLAHE (#24) e dos experimentos E1-E4 (#13-#17).\n"
       "Antes de rodar: **Ambiente de execução -> Alterar tipo de ambiente de execução -> GPU**.\n\n"
       "Pré-requisito único feito uma vez na sua máquina: subir um zip de `data/A. Segmentation/` "
       "para o Google Drive."),
    md("## 1. Montar o Drive e clonar o repositório"),
    code("from google.colab import drive\n"
         "drive.mount('/content/drive')\n\n"
         "!git clone https://github.com/meowaax/pdi-diabetic-retinopathy.git\n"
         "%cd pdi-diabetic-retinopathy"),
    md("## 2. Instalar dependências e preparar os dados\n\n"
       "Ajuste o caminho do zip no Drive se for diferente."),
    code("!pip install -q -r requirements.txt\n\n"
         "ZIP_NO_DRIVE = '/content/drive/MyDrive/A_Segmentation.zip'  # ajuste se necessário\n"
         "!mkdir -p data\n"
         "!unzip -q \"$ZIP_NO_DRIVE\" -d data/\n"
         "!ls \"data/A. Segmentation\""),
    code("!python scripts/build_cache.py"),
    md("## 3. (opcional) Confirmar que os testes passam no ambiente do Colab"),
    code("!python -m pytest -q tests -x"),
    md("## 4. Apontar checkpoints/logs para o Drive (sobrevive ao fim da sessão)"),
    code("import os\n"
         "RESULTS_NO_DRIVE = '/content/drive/MyDrive/pdi_resultados'\n"
         "os.makedirs(f'{RESULTS_NO_DRIVE}/checkpoints', exist_ok=True)\n"
         "os.makedirs(f'{RESULTS_NO_DRIVE}/logs', exist_ok=True)\n"
         "os.makedirs(f'{RESULTS_NO_DRIVE}/runs', exist_ok=True)"),
    md("## 4b. Função auxiliar: roda um comando mostrando a saída em tempo real\n\n"
       "`!comando | tail -N` só mostra algo **depois** que o processo termina, o que parece "
       "travado em treinos longos. `run_live` imprime cada linha (do loop de treino, uma por "
       "época) assim que ela é gerada, com hora e um prefixo identificando a run."),
    code("import subprocess\n"
         "import sys\n"
         "import time\n\n"
         "def run_live(cmd: list[str], prefix: str = \"\") -> int:\n"
         "    \"\"\"Executa cmd (lista de argumentos) imprimindo stdout/stderr linha a linha, "
         "em tempo real, com timestamp. Devolve o código de saída.\"\"\"\n"
         "    t0 = time.time()\n"
         "    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
         "                             text=True, bufsize=1)\n"
         "    for line in proc.stdout:\n"
         "        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))\n"
         "        print(f\"[{elapsed}]{prefix} {line}\", end='')\n"
         "        sys.stdout.flush()\n"
         "    proc.wait()\n"
         "    return proc.returncode"),
    md("## 5. Sensibilidade do CLAHE (issue #24): as 8 configurações\n\n"
       "Cada treino usa a divisão de desenvolvimento (42 treino / 12 validação). "
       "Roda uma configuração por vez, mostrando o progresso época a época; "
       "se a sessão cair, repita a célula — ela pula as configurações que já têm resultado salvo.\n\n"
       "**Nota:** o log deve mostrar uma linha `mixed_precision=True ignorado: ...`. Isso é "
       "esperado: mixed precision foi desativado porque causou `loss=nan` de forma persistente "
       "em testes (ver `src/train.py`). O treino roda em precisão total (mais lento, mas correto)."),
    code("import glob\n"
         "configs = sorted(glob.glob('configs/clahe_grid/*.yaml'))\n"
         "print(f'{len(configs)} configurações:')\n"
         "configs"),
    code("import os\n\n"
         "for idx, cfg_path in enumerate(configs, start=1):\n"
         "    run_name = cfg_path.split('/')[-1].replace('.yaml', '')\n"
         "    out_json = f'results/runs/{run_name}.json'\n"
         "    prefixo = f' ({idx}/{len(configs)} {run_name})'\n"
         "    if os.path.exists(out_json):\n"
         "        print(f'== {idx}/{len(configs)} {run_name}: já existe, pulando ==')\n"
         "        continue\n"
         "    print(f'== {idx}/{len(configs)} {run_name}: iniciando ==')\n"
         "    rc = run_live(\n"
         "        ['python', 'scripts/run_experiment.py', cfg_path, '--subset', '--run-name', run_name],\n"
         "        prefix=prefixo,\n"
         "    )\n"
         "    print(f'== {idx}/{len(configs)} {run_name}: {\"OK\" if rc == 0 else f\"FALHOU (codigo {rc})\"} ==')\n"
         "    !cp -r results/checkpoints results/logs results/runs \"$RESULTS_NO_DRIVE/\""),
    md("## 6. Conferir a sensibilidade do CLAHE"),
    code("!python scripts/collect_results.py --prefix clahe_"),
    md("## 7. E1-E4: validação cruzada em 5 dobras (issues #13-#17)\n\n"
       "Rodar **depois** de decidir o preprocess_mode de cada experimento "
       "(E1=rgb, E2=clahe com os parâmetros congelados na #24, E3=green, E4=clahe_green opcional) "
       "e os hiperparâmetros finais em `configs/train_default.yaml`.\n\n"
       "Exemplo para o E1 (RGB), uma dobra por vez:"),
    code("import os\n\n"
         "EXPERIMENTO = 'e1_rgb'       # troque para e2_clahe / e3_green / e4_clahe_green\n"
         "CONFIG = 'configs/train_default.yaml'  # ou um YAML específico do experimento\n\n"
         "for k in range(5):\n"
         "    run_name = f'{EXPERIMENTO}_fold{k}'\n"
         "    out_json = f'results/runs/{run_name}.json'\n"
         "    prefixo = f' (dobra {k+1}/5 {run_name})'\n"
         "    if os.path.exists(out_json):\n"
         "        print(f'== dobra {k+1}/5 {run_name}: já existe, pulando ==')\n"
         "        continue\n"
         "    print(f'== dobra {k+1}/5 {run_name}: iniciando ==')\n"
         "    rc = run_live(\n"
         "        ['python', 'scripts/run_experiment.py', CONFIG, '--fold', str(k), '--run-name', run_name],\n"
         "        prefix=prefixo,\n"
         "    )\n"
         "    print(f'== dobra {k+1}/5 {run_name}: {\"OK\" if rc == 0 else f\"FALHOU (codigo {rc})\"} ==')\n"
         "    !cp -r results/checkpoints results/logs results/runs \"$RESULTS_NO_DRIVE/\""),
    md("## 8. Agregar os resultados finais (média ± desvio entre dobras)"),
    code("!python scripts/collect_results.py"),
    md("## 9. Salvar tudo no Drive para devolver depois\n\n"
       "`results/runs/*.json` é o essencial (pequeno, texto). Os checkpoints (`.pt`) são grandes; "
       "mantenha-os só se for continuar o treino depois, senão descarte para economizar espaço."),
    code("!cp -r results/summaries \"$RESULTS_NO_DRIVE/\"\n"
         "!ls -la \"$RESULTS_NO_DRIVE\""),
]
nbf.write(nb, "notebooks/colab_runner.ipynb")
print("gerado: notebooks/colab_runner.ipynb")
