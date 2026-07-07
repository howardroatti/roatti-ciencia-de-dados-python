"""Utilidades compartilhadas do código companheiro do livro
*Ciência de Dados com Python* (Howard Roatti). Licença MIT.
"""
import os

# Caminho absoluto para a pasta de dados, independente de onde o script é chamado.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def caminho_dados(*partes):
    """Retorna o caminho absoluto de um arquivo dentro de ``data/``."""
    return os.path.join(DATA_DIR, *partes)
