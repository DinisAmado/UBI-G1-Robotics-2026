#main.py - O "Cérebro" do Unitree G1 
import vision_module       # Grupo 3
import navigation_module   # Grupo 2
import grasping_module     # Grupo 4
import hmi_module          # Grupo 5 
import slam_module         # Grupo 2 

def missao_principal():
    #Passo 1 (SLAM): Guardar a posição da pessoa 
    posicao_pessoa = slam_module.get_current_robot_pose() 
    
    #Passo 2 (HMI): Esperar por comando de voz 
    comando = hmi_module.wait_for_command() 
    alvo_nome = hmi_module.parse_intent(comando) 

    #Passo 3 (Vision): Localizar o objeto pedido 
    target_data = vision_module.detect_object(alvo_nome)
    
    if target_data['confidence'] > 0.8:
        coords_obj = target_data['position']
        
        #Passo 4 (Navigation & Grasping): Ir até à mesa e agarrar 
        if navigation_module.go_to(coords_obj[0], coords_obj[1]):
            success = grasping_module.pick_up(target_data['object_id'])
            
            if success:
                #Passo 5 (Navigation): Movimentar até à pessoa
                print("Orquestrador: Objeto seguro. Regressando ao utilizador...")
                if navigation_module.go_to(posicao_pessoa['x'], posicao_pessoa['y']):
                    
                    #Passo 6 (Grasping): Entregar o objeto 
                    grasping_module.deliver() 
                    hmi_module.say("Aqui está o seu objeto. Posso ajudar em algo mais?")
                else:
                    hmi_module.say("Cheguei perto, mas não consigo aproximar-me mais para entregar.")
            else:
                hmi_module.say("Tentei agarrar o objeto, mas ele escorregou.")
    else:
        hmi_module.say("Desculpe, não consegui localizar o objeto na mesa.")

if __name__ == "__main__":
    missao_principal()