"""
MAIN_HMI.py

Grupo 5 — HMI (rt/hmi/*)

Este script publica intenções para a orquestração no tópico:
    rt/hmi/intent

E lê feedback da orquestração no tópico:
    rt/hmi/feedback

Deve estar na mesma pasta que:
    - idl_ri.py
    - qos_profiles.py
"""

import time
import logging
from typing import Optional

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from qos_profiles import QOS_HMI

from idl_ri import (
    Header,
    Intent,
    Acao,
    Feedback,
)


# ==============================================================================
# CONFIGURAÇÃO DDS
# ==============================================================================

DOMAIN_ID = 0

TOPIC_INTENT = "rt/hmi/intent"
TOPIC_FEEDBACK = "rt/hmi/feedback"

FRAME_ID = "hmi"


# ==============================================================================
# INICIALIZAÇÃO DDS
# ==============================================================================

dp = DomainParticipant(DOMAIN_ID)
pub = Publisher(dp)
sub = Subscriber(dp)

t_intent = Topic(
    dp,
    TOPIC_INTENT,
    Intent,
    qos=QOS_HMI
)

t_feedback = Topic(
    dp,
    TOPIC_FEEDBACK,
    Feedback,
    qos=QOS_HMI
)

w_intent = DataWriter(pub, t_intent)
r_feedback = DataReader(sub, t_feedback)


# ==============================================================================
# CONTADOR DE MENSAGENS
# ==============================================================================

seq = 0


# ==============================================================================
# MAPEAMENTO ENTRE A NOSSA HRI E A ORQUESTRAÇÃO
# ==============================================================================

"""
A nossa HRI usa ações como:
    TRAZER
    IR_BUSCAR
    AGARRAR
    LARGAR
    PARAR

A orquestração espera o enum Acao:
    ENTREGAR
    RECOLHER
    SEGUIR
    PARAR
    LARGA
"""

HRI_TO_ORCH_ACTION = {
    "TRAZER": Acao.ENTREGAR,
    "IR_BUSCAR": Acao.RECOLHER,
    "AGARRAR": Acao.RECOLHER,
    "LARGAR": Acao.LARGA,
    "PARAR": Acao.PARAR,
}


"""
A nossa HRI usa targets como:
    BOLA_DE_TENIS
    CUBO_DE_RUBIK
    PASTA_DE_DENTES

A orquestração recebe o alvo como string.
"""

HRI_TARGET_TO_ALVO = {
    "BOLA_DE_TENIS": "bola_de_tenis",
    "CUBO_DE_RUBIK": "cubo_de_rubik",
    "PASTA_DE_DENTES": "pasta_de_dentes",
    "NENHUM": "",
    "DESCONHECIDO": "",
}


# ==============================================================================
# PUBLICAR INTENT PARA A ORQUESTRAÇÃO
# ==============================================================================

def send_intent(acao: Acao, alvo: str, comando_grasping: str = ""):
    """
    Publica uma Intent diretamente para a orquestração.

    Parâmetros:
        acao:
            Enum Acao definido em idl_ri.py.
            Exemplo: Acao.ENTREGAR, Acao.RECOLHER, Acao.PARAR

        alvo:
            Objeto ou pessoa alvo.
            Exemplo: "bola_de_tenis", "cubo_de_rubik", "pasta_de_dentes"

        comando_grasping:
            Comando opcional para o módulo de grasping.
            Exemplo: "agarrar"
    """

    global seq
    seq += 1

    msg = Intent(
        header=Header(
            timestamp_ns=time.time_ns(),
            frame_id=FRAME_ID,
            seq=seq,
        ),
        acao=acao,
        alvo=alvo,
        comando_grasping=comando_grasping,
    )

    w_intent.write(msg)

    logging.getLogger("hmi").info(
        "[INTENT ENVIADA] acao=%s alvo=%s grasping=%s seq=%d",
        acao.name,
        alvo,
        comando_grasping,
        seq,
    )


# ==============================================================================
# PUBLICAR INTENT A PARTIR DA NOSSA HRI
# ==============================================================================

def send_hri_intent(hri_action: str, hri_target: str):
    """
    Converte a ação/alvo vindos da nossa HRI para o formato esperado
    pela orquestração e publica no tópico rt/hmi/intent.

    Exemplo:
        HRI:
            action = "TRAZER"
            target = "BOLA_DE_TENIS"

        Orquestração:
            acao = Acao.ENTREGAR
            alvo = "bola_de_tenis"
            comando_grasping = "agarrar"
    """

    log = logging.getLogger("hmi")

    if hri_action not in HRI_TO_ORCH_ACTION:
        log.warning(
            "Ação HRI não suportada pela orquestração: %s",
            hri_action,
        )
        return False

    acao = HRI_TO_ORCH_ACTION[hri_action]
    alvo = HRI_TARGET_TO_ALVO.get(hri_target, "")

    comando_grasping = ""

    if hri_action in ["TRAZER", "IR_BUSCAR", "AGARRAR"]:
        comando_grasping = "agarrar"

    send_intent(
        acao=acao,
        alvo=alvo,
        comando_grasping=comando_grasping,
    )

    return True


# ==============================================================================
# LER FEEDBACK DA ORQUESTRAÇÃO
# ==============================================================================

def poll_feedback() -> Optional[Feedback]:
    """
    Lê uma mensagem de feedback da orquestração, se existir.

    Retorna:
        Feedback, se houver mensagem;
        None, se não houver feedback novo.
    """

    samples = r_feedback.take(1)

    if samples:
        return samples[0]

    return None


# ==============================================================================
# LOOP PRINCIPAL DE TESTE
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("hmi")

    log.info("HMI a iniciar no domínio %d", DOMAIN_ID)
    log.info("A publicar em: %s", TOPIC_INTENT)
    log.info("A escutar feedback em: %s", TOPIC_FEEDBACK)

    # --------------------------------------------------------------------------
    # TESTE MANUAL
    # --------------------------------------------------------------------------
    # Este teste simula uma frase da nossa HRI:
    # "traz-me a bola de ténis"
    #
    # A nossa HRI produziria:
    # action = "TRAZER"
    # target = "BOLA_DE_TENIS"
    #
    # Este script converte isso para:
    # Acao.ENTREGAR
    # alvo = "bola_de_tenis"
    # comando_grasping = "agarrar"
    # --------------------------------------------------------------------------

    send_hri_intent("TRAZER", "BOLA_DE_TENIS")

    # Outros testes possíveis:
    # send_hri_intent("IR_BUSCAR", "CUBO_DE_RUBIK")
    # send_hri_intent("AGARRAR", "PASTA_DE_DENTES")
    # send_hri_intent("PARAR", "NENHUM")
    # send_hri_intent("LARGAR", "NENHUM")

    while True:
        fb = poll_feedback()

        if fb:
            log.info(
                "[FEEDBACK] estado=%s msg=%s",
                fb.state.name,
                fb.message,
            )

        time.sleep(0.1)
