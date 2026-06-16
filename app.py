# app.py
# Backend Flask para a interface de predição de LOD
#
# Requisitos:
#   pip install flask flask-cors joblib numpy pandas scikit-learn --break-system-packages
#
# Como executar:
#   1. Copie a pasta "modelos/" (gerada pelo notebook, contendo pipeline_completo.joblib)
#      para o mesmo diretório deste arquivo app.py
#   2. python app.py
#   3. Abra index.html no navegador (ele faz requisições para http://localhost:5000)

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# 1. ANALITOS (mesmo dicionário do notebook)
# ---------------------------------------------------------------------------
ANALITOS = {
    "ciprofloxacin": {"nome": "Ciprofloxacin", "pKa1": 6.09, "pKa2": 8.74, "MolWt": 331.34, "NumHBD": 2, "NumHBA": 5},
    "enrofloxacin":  {"nome": "Enrofloxacin",  "pKa1": 6.09, "pKa2": 7.70, "MolWt": 359.40, "NumHBD": 1, "NumHBA": 5},
    "levofloxacin":  {"nome": "Levofloxacin",  "pKa1": 5.67, "pKa2": 8.00, "MolWt": 361.37, "NumHBD": 1, "NumHBA": 6},
    "norfloxacin":   {"nome": "Norfloxacin",   "pKa1": 6.30, "pKa2": 8.40, "MolWt": 319.33, "NumHBD": 2, "NumHBA": 5},
    "ofloxacin":     {"nome": "Ofloxacin",     "pKa1": 5.67, "pKa2": 8.00, "MolWt": 361.37, "NumHBD": 1, "NumHBA": 6},
    "moxifloxacin":  {"nome": "Moxifloxacin",  "pKa1": 6.40, "pKa2": 9.50, "MolWt": 401.43, "NumHBD": 2, "NumHBA": 6},
    "lomefloxacin":  {"nome": "Lomefloxacin",  "pKa1": 5.74, "pKa2": 8.10, "MolWt": 351.35, "NumHBD": 2, "NumHBA": 4},
    "gatifloxacin":  {"nome": "Gatifloxacin",  "pKa1": 6.00, "pKa2": 8.74, "MolWt": 375.40, "NumHBD": 2, "NumHBA": 5},
    "pefloxacin":    {"nome": "Pefloxacin",    "pKa1": 6.02, "pKa2": 7.80, "MolWt": 333.36, "NumHBD": 1, "NumHBA": 4},
    "sparfloxacin":  {"nome": "Sparfloxacin",  "pKa1": 6.30, "pKa2": 9.00, "MolWt": 392.41, "NumHBD": 3, "NumHBA": 5},
}

# ---------------------------------------------------------------------------
# 2. FEATURES COMPLETAS NA ORDEM DO TREINAMENTO
# ---------------------------------------------------------------------------
FEATURES_COMPLETAS = [
    'pH', 'pH-pKa1', 'pH-pKa2',
    'RE_AgAgCl', 'RE_SCE',
    'Method_CV', 'Method_DPV', 'Method_LSV', 'Method_SWV', 'Method_stripping',
    'Substrate_Au', 'Substrate_BDD', 'Substrate_CPE', 'Substrate_GCE',
    'Substrate_GS', 'Substrate_PGE', 'Substrate_SPE',
    'num_carbon', 'num_qds', 'num_org', 'num_mip', 'num_inorg',
    'num_surf', 'num_mof_cof', 'num_nano', 'num_oxidenano', 'num_components',
    'MolWt', 'NumHBD', 'NumHBA', 'log_ConcEle'
]

# ---------------------------------------------------------------------------
# 3. CARREGAMENTO DO PIPELINE
# ---------------------------------------------------------------------------
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'pipeline_completo.joblib')

# Opcional: caminho de um modelo Ridge salvo separadamente. Se existir, ele
# substitui o "best_model" do pipeline_completo.joblib para a predição.
# O notebook original compara LR / RF / GBM / XGB; se você treinar e salvar
# um RidgeClassifier separadamente (ex.: joblib.dump(ridge_model,
# 'modelos/ridge.joblib')), coloque o arquivo nesta pasta e ele será usado.
RIDGE_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'ridge.joblib')

_pipeline = None


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Pipeline não encontrado em: {PIPELINE_PATH}\n"
            "Copie a pasta 'modelos/' gerada pelo notebook para junto deste app.py."
        )

    data = joblib.load(PIPELINE_PATH)

    model = data.get('best_model')
    model_name = data.get('best_model_name', 'best_model')

    if os.path.exists(RIDGE_PATH):
        model = joblib.load(RIDGE_PATH)
        model_name = 'Ridge'

    _pipeline = {
        'model': model,
        'model_name': model_name,
        'imputer': data.get('imputer'),
        'scaler': data.get('scaler'),
        'selector': data.get('selector'),
        'feature_cols': data.get('feature_cols', FEATURES_COMPLETAS),
        'selected_features': data.get('selected_features'),
        'threshold_lod': data.get('threshold_lod', 0.01),
        'decision_threshold': data.get('decision_threshold', 0.5),
    }
    return _pipeline


# ---------------------------------------------------------------------------
# 4. FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------
def calcular_features_analito(pH, analito_key):
    dados = ANALITOS.get(analito_key, ANALITOS["ciprofloxacin"])
    return {
        "pH-pKa1": pH - dados["pKa1"],
        "pH-pKa2": pH - dados["pKa2"],
        "MolWt": dados["MolWt"],
        "NumHBD": dados["NumHBD"],
        "NumHBA": dados["NumHBA"],
        "nome_analito": dados["nome"],
    }


def montar_condicoes(payload):
    """Recebe o JSON do front-end e monta o dicionário completo de features."""
    condicoes = {}

    analito_key = payload.get('analito', 'ciprofloxacin')
    pH = float(payload.get('pH', 7.0))
    conc_ele = float(payload.get('ConcEleM', 0.1))

    condicoes['pH'] = pH
    condicoes['log_ConcEle'] = float(np.log10(conc_ele + 1e-12))

    # Eletrodo de referência
    re_sel = payload.get('RE', 'AgAgCl')
    condicoes['RE_AgAgCl'] = 1 if re_sel == 'AgAgCl' else 0
    condicoes['RE_SCE'] = 1 if re_sel == 'SCE' else 0

    # Método voltamétrico
    metodo_sel = payload.get('Method', 'DPV')
    for m in ['Method_CV', 'Method_DPV', 'Method_LSV', 'Method_SWV', 'Method_stripping']:
        condicoes[m] = 0
    condicoes[f'Method_{metodo_sel}'] = 1

    # Substrato
    substrato_sel = payload.get('Substrate', 'GCE')
    for s in ['Substrate_Au', 'Substrate_BDD', 'Substrate_CPE', 'Substrate_GCE',
              'Substrate_GS', 'Substrate_PGE', 'Substrate_SPE']:
        condicoes[s] = 0
    condicoes[f'Substrate_{substrato_sel}'] = 1

    # Modificadores
    for key in ['num_mip', 'num_nano', 'num_carbon', 'num_oxidenano',
                 'num_org', 'num_inorg', 'num_qds', 'num_surf', 'num_mof_cof']:
        condicoes[key] = int(payload.get(key, 0))

    condicoes['num_components'] = sum(
        condicoes[k] for k in ['num_mip', 'num_nano', 'num_carbon', 'num_oxidenano',
                                'num_org', 'num_inorg', 'num_qds', 'num_surf', 'num_mof_cof']
    )

    # Propriedades do analito
    auto = calcular_features_analito(pH, analito_key)
    condicoes['pH-pKa1'] = auto['pH-pKa1']
    condicoes['pH-pKa2'] = auto['pH-pKa2']
    condicoes['MolWt'] = auto['MolWt']
    condicoes['NumHBD'] = auto['NumHBD']
    condicoes['NumHBA'] = auto['NumHBA']

    return condicoes, auto['nome_analito']


def prever(condicoes, pl):
    df = pd.DataFrame([{feat: condicoes.get(feat, 0) for feat in pl['feature_cols']}]).fillna(0)

    X_imp = pl['imputer'].transform(df)
    X_scaled = pl['scaler'].transform(X_imp)
    X_sel = pl['selector'].transform(X_scaled)

    proba = pl['model'].predict_proba(X_sel)[0][1]
    thr = pl['decision_threshold']
    classe = int(proba >= thr)

    return {
        'probabilidade_lod_baixo': float(proba),
        'probabilidade_lod_alto': float(1 - proba),
        'classe': classe,
        'classe_label': f"LOD ≤ {pl['threshold_lod']} µM" if classe == 1 else f"LOD > {pl['threshold_lod']} µM",
        'threshold_lod': pl['threshold_lod'],
        'decision_threshold': thr,
        'modelo': pl['model_name'],
    }


# ---------------------------------------------------------------------------
# 5. ROTAS
# ---------------------------------------------------------------------------
@app.route('/api/options', methods=['GET'])
def options():
    """Retorna as opções disponíveis para popular os selects do front-end."""
    return jsonify({
        'analitos': {k: v['nome'] for k, v in ANALITOS.items()},
        'metodos': ['CV', 'DPV', 'LSV', 'SWV', 'stripping'],
        'eletrodos_referencia': ['AgAgCl', 'SCE'],
        'substratos': ['GCE', 'CPE', 'SPE', 'GS', 'Au', 'BDD', 'PGE'],
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        pl = load_pipeline()
        payload = request.get_json(force=True)
        condicoes, nome_analito = montar_condicoes(payload)
        resultado = prever(condicoes, pl)
        resultado['analito'] = nome_analito
        resultado['condicoes'] = condicoes
        return jsonify({'ok': True, 'resultado': resultado})
    except Exception as e:
        return jsonify({'ok': False, 'erro': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    try:
        pl = load_pipeline()
        return jsonify({'ok': True, 'modelo': pl['model_name']})
    except Exception as e:
        return jsonify({'ok': False, 'erro': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
