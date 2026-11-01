# -*- coding: utf-8 -*-
"""Dados FICTÍCIOS dos Módulos VIII e IX (escritos/gerados por código).

Tudo abaixo é inventado pelo autor apenas para ilustrar as técnicas do livro:
a "Mercado Aurora" é uma empresa inexistente; seus documentos de política, as
perguntas e os vetores de *embedding* não vêm de nenhuma base real, de terceiros
ou de repositórios públicos. Nada aqui representa regras de uma empresa de verdade.

Funções:
    corpus_bpe()               -> corpus minúsculo para o BPE de brinquedo (Módulo VIII)
    palavras_embeddings()      -> vetores 4D fictícios com eixos interpretáveis (Módulo VIII)
    documentos_ficticios()     -> dict {id: texto} com a base da Mercado Aurora (Módulo IX)
    metadados_documentos()     -> dict {id: {"titulo", "vigencia"}} (Módulo IX)
    perguntas_ficticias()      -> lista de (pergunta, id_do_documento_correto)
    perguntas_fora_da_base()   -> perguntas cuja resposta NÃO está nos documentos
"""
import numpy as np


# --------------------------------------------------------------------- Módulo VIII
def corpus_bpe():
    """Corpus de brinquedo: poucas palavras repetidas, para o BPE aprender fusões."""
    return ["baixo", "baixo", "baixo", "baixa", "baixa", "caixa", "caixa", "caixas",
            "peixe", "peixe", "peixes"]


def palavras_embeddings():
    """Embeddings 4D inventados. Eixos (só para interpretação): realeza, gênero
    (+ = feminino), humano, animal. Não vêm de nenhum modelo treinado."""
    return {
        "rei":      np.array([0.9, -0.8, 0.9, 0.0]),
        "rainha":   np.array([0.9,  0.8, 0.9, 0.0]),
        "homem":    np.array([0.0, -0.8, 0.9, 0.0]),
        "mulher":   np.array([0.0,  0.8, 0.9, 0.0]),
        "gato":     np.array([0.0,  0.1, 0.0, 0.9]),
        "felino":   np.array([0.0,  0.0, 0.0, 0.9]),
        "cachorro": np.array([0.0, -0.1, 0.0, 0.8]),
        "bolsa":    np.array([0.1,  0.0, -0.5, 0.05]),
    }


# --------------------------------------------------------------------- Módulo IX
def documentos_ficticios():
    """Base de conhecimento fictícia da 'Mercado Aurora' (loja online inexistente)."""
    return {
        "reembolso": (
            "Política de reembolso. O cliente pode devolver qualquer produto em até 30 dias "
            "corridos após o recebimento, desde que a embalagem original esteja preservada. "
            "O valor é devolvido na mesma forma de pagamento usada na compra. Pagamentos no "
            "cartão de crédito são estornados em até duas faturas. Pagamentos via Pix são "
            "devolvidos em até 5 dias úteis depois da aprovação da devolução."),
        "reembolso_antigo": (
            "Política de reembolso (versão antiga, revogada em 2023). O cliente podia devolver "
            "produtos em até 15 dias corridos após o recebimento. O valor era devolvido "
            "somente na forma de crédito na loja. Esta regra não está mais em vigor e foi "
            "substituída pela política de reembolso vigente."),
        "promocao": (
            "Produtos em promoção. Itens vendidos com desconto de liquidação não podem ser "
            "trocados por outro modelo, apenas reembolsados dentro do prazo normal de devolução. "
            "Cupons de desconto não são acumulativos: só um cupom pode ser usado por pedido. "
            "O frete grátis vale para compras acima de 200 reais em todo o território nacional."),
        "entrega": (
            "Prazos de entrega. Pedidos para capitais chegam em até 5 dias úteis, e para o "
            "interior em até 10 dias úteis, contados a partir da confirmação do pagamento. "
            "O código de rastreio é enviado por e-mail assim que o pacote sai do centro de "
            "distribuição. Se ninguém estiver em casa, a transportadora faz até duas "
            "tentativas de entrega antes de devolver o pacote à loja."),
        "garantia": (
            "Garantia dos produtos. Eletrônicos têm garantia de 12 meses contra defeitos de "
            "fabricação, contados da data da nota fiscal. Móveis têm garantia de 6 meses. "
            "A garantia não cobre mau uso, quedas ou contato com líquidos. Para acionar a "
            "garantia, o cliente deve abrir um chamado no site anexando a nota fiscal e fotos "
            "do defeito."),
        "atendimento": (
            "Horário de atendimento. A central funciona de segunda a sexta, das 9h às 18h, e "
            "aos sábados, das 9h às 13h. Não há atendimento aos domingos e feriados nacionais. "
            "O chat do site responde em até 10 minutos no horário comercial. Reclamações "
            "enviadas por e-mail são respondidas em até 2 dias úteis."),
        "fidelidade": (
            "Programa de fidelidade Aurora Pontos. A cada 1 real gasto o cliente ganha 1 ponto. "
            "Cada 100 pontos valem um desconto de 5 reais na compra seguinte. Os pontos expiram "
            "12 meses após o crédito. Pontos não são devolvidos em caso de cancelamento do "
            "pedido, e o desconto não vale para a compra de vales-presente."),
        "pagamento": (
            "Formas de pagamento. Aceitamos cartão de crédito em até 10 parcelas sem juros para "
            "compras acima de 300 reais, boleto bancário e Pix. O boleto vence em 3 dias e o "
            "pedido só é separado após a compensação. O Pix tem 5 por cento de desconto no "
            "valor total. Não aceitamos cheques nem pagamento em dinheiro na entrega."),
        "privacidade": (
            "Privacidade e dados pessoais. Os dados dos clientes são usados apenas para processar "
            "pedidos, emitir notas fiscais e enviar comunicações autorizadas. O cliente pode "
            "pedir a exclusão dos seus dados a qualquer momento, pelo e-mail do encarregado de "
            "dados. Não vendemos dados a terceiros. Cookies de publicidade só são ativados "
            "mediante consentimento."),
    }


def metadados_documentos():
    """Metadados de cada documento (título e se a regra está vigente)."""
    titulos = {
        "reembolso": "Política de reembolso", "reembolso_antigo": "Reembolso (versão antiga)",
        "promocao": "Produtos em promoção", "entrega": "Prazos de entrega",
        "garantia": "Garantia", "atendimento": "Atendimento", "fidelidade": "Fidelidade",
        "pagamento": "Formas de pagamento", "privacidade": "Privacidade",
    }
    return {k: {"titulo": v, "vigencia": "revogada" if k == "reembolso_antigo" else "vigente"}
            for k, v in titulos.items()}


def perguntas_ficticias():
    """(pergunta, id do documento que contém a resposta). Misturam perguntas com muita
    sobreposição de palavras e paráfrases com pouca sobreposição."""
    return [
        ("Em quantos dias posso devolver um produto?", "reembolso"),
        ("Como recebo meu dinheiro de volta se paguei com Pix?", "reembolso"),
        ("Posso trocar um item em liquidação por outro modelo?", "promocao"),
        ("Dá para usar dois cupons no mesmo pedido?", "promocao"),
        ("Quanto tempo demora a entrega para o interior?", "entrega"),
        ("Quando recebo o código de rastreio?", "entrega"),
        ("Quantas tentativas de entrega a transportadora faz?", "entrega"),
        ("Qual a garantia de um notebook?", "garantia"),
        ("A garantia cobre queda do aparelho?", "garantia"),
        ("A central abre aos sábados?", "atendimento"),
        ("Em que horário o chat responde?", "atendimento"),
        ("Quantos pontos ganho a cada real gasto?", "fidelidade"),
        ("Os pontos de fidelidade vencem?", "fidelidade"),
        ("Posso parcelar a compra em 10 vezes sem juros?", "pagamento"),
        ("O boleto tem prazo de vencimento?", "pagamento"),
        ("Posso pedir para apagarem meus dados?", "privacidade"),
        ("Vocês repassam minhas informações para outras empresas?", "privacidade"),
        ("Quanto de desconto tenho pagando via Pix?", "pagamento"),
        # paráfrases com pouquíssimas palavras em comum com o texto (difíceis para busca lexical)
        ("Posso dividir o pagamento no cartão?", "pagamento"),
        ("O que acontece se eu não estiver em casa na hora da entrega?", "entrega"),
        ("Por quanto tempo posso usar o produto com defeito sem pagar conserto?", "garantia"),
        ("Dá para ser atendido num domingo?", "atendimento"),
        ("Meus dados podem ser deletados?", "privacidade"),
        ("Qual o valor mínimo para não pagar frete?", "promocao"),
    ]


def perguntas_fora_da_base():
    """Perguntas cuja resposta não existe na base (o sistema deveria recusar)."""
    return [
        "Qual é o CNPJ da empresa?",
        "Vocês têm loja física em Curitiba?",
        "Qual a capital da Austrália?",
        "Como faço para trabalhar na empresa?",
    ]
