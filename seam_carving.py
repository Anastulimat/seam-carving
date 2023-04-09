# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import sys
import math

'''
Première étape : Détermination de la carte d'énergie de l'image (gradient) en utilisant un dual gradient.

Cette fonction permet de déterminer et afficher le gradient de l'image donnée.
Nous avons utilisé les fonctions décrites dans ce site que nous l'avons trouvé intéressant :
https://www.datasciencecentral.com/profiles/blogs/seam-carving-using-dynamic-programming-to-implement-context-aware
'''
def gradient(image):
  image1 = image.load()
  image2 = Image.new('L', (image.size[0], image.size[1]))
  image3 = image2.load()

  for x in range(image.size[0]):
    for y in range(image.size[1]):
      xpl = (x-1) % image.size[0]
      xpr = (x+1) % image.size[0]
      ypu = (y-1) % image.size[1]
      ypd = (y+1) % image.size[1]
      drx = image1[xpl, y][0] - image1[xpr, y][0]
      dgx = image1[xpl, y][1] - image1[xpr, y][1]
      dbx = image1[xpl, y][2] - image1[xpr, y][2]
      dry = image1[x, ypu][0] - image1[x, ypd][0]
      dgy = image1[x, ypu][1] - image1[x, ypd][1]
      dby = image1[x, ypu][2] - image1[x, ypd][2]
      G = round(math.sqrt(math.pow(drx, 2) + math.pow(dgx, 2) + math.pow(dbx, 2) + math.pow(dry, 2) + math.pow(dgy, 2) + math.pow(dby, 2)))
      image3[x,y] = (G,)

  return image2


'''
Deuxième étape : Détermination de la seam optimal.

Cette fonction calcule une matrice de coût d'une image à partir de son gradient
'''
def calculate_cost_matrix(gradient_image):
    image = gradient_image
    cost_matrix = [[0 for i in range(image.size[0])] for j in range(image.size[1])]
    pixels = image.load()

    # Ici on calcule la matrice des coûts
    for y in range(0, image.size[1]):
        if y > 0:
            last_line = cost_matrix[y-1]
            
        #On crée un nouveau tableau pour chaque ligne du tableau
        for x in range(0, image.size[0]):

            if y > 0:
                if x > 0 and x < image.size[0] - 1:
                    #Dans ce cas là, on se retrouve au milieu et donc le cout du pixel est la somme de la valeur du pixel
                    #avec le minimum du cout du pixel en haut et celui en haut à droite et celui en haut à gauche
                    cost = pixels[x, y] + min(last_line[x-1], last_line[x], last_line[x+1])

                elif x <= 0:
                    #Dans ce cas là on est à gauche de l'image et donc le cout du pixel est la somme 
                    #de la valeur du pixel avec le minimum du cout du pixel en haut et celui en haut à droite
                    cost = pixels[x, y] + min(last_line[x], last_line[x+1])

                elif x >= image.size[0]-1:
                    #Dans ce cas là on est à droite de l'image et donc le cout du pixel est la somme 
                    #de la valeur du pixel avec le minimum du cout du pixel en haut et celui en haut à gauche
                    cost = pixels[x, y] + min(last_line[x-1], last_line[x])

            else:
                cost = pixels[x, y]

            #Puis on rajoute le cout du pixel
            cost_matrix[y][x] = cost

    return cost_matrix
    
'''
Deuxième étape : Détermination de la seam optimal.

Cette fonction détermine la seam optimale, qui sera enlevée, à partir de la matrice des coûts
'''
def sema_detection(cost_matrix):
    seam = []
    y = len(cost_matrix)-1

    # dans la ligne tout en bas de l'image 
    # On cherche la valeur minimum sur toute la ligne
    x = int(np.argmin(cost_matrix[y]))
    # On rajoute au seam le tuple du pixel à supprimer sur la ligne du bas
    seam += [(x, y)]

    # Pour toutes les autres lignes que celle du bas on va remonter
    for y in range(len(cost_matrix)-2, -1, -1):

        if x-1 < 0:
            # on est à gauche de l'image
            x = int(np.argmin(cost_matrix[y][x:x + 2])) + x
                    
        elif x+1 >= len(cost_matrix[0]):
            # on est à droite de l'image
            x = int(np.argmin(cost_matrix[y][x - 1:x + 1])) + x - 1
            
        else:
            # on est au milieu de l'image
            x = int(np.argmin(cost_matrix[y][x - 1:x + 2])) + x - 1

        #  une fois ce minimum au niveau des couts calculé, on ajoute cela au seam
        seam += [(x, y)]
    # une fois toutes les lignes traitées on peut retourner la liste de tuples contenant les poins à supprimer
    return seam


'''
Troisième étape : Destruction de la seam déterminée et décalage de l'image

Cette fonction permet de supprimer la seam optimale calculée, d'abord on suprrime la seam en décalant les pixels par dessus 
et puis on redimensionnemene l'image
'''
def delete_seam(im, seam):
    image = im.load()
    for element in seam: #pour chaque element du seam (de la forme (x,y))
        for x in range (element[0], im.size[0]-1): 
            image[x, element[1]] = image[x+1, element[1]]

    boundries = (0, 0, im.size[0]-1, im.size[1])
    result_image = im.crop(boundries)    
    #On retourne l'image modifiée
    return  result_image


'''
Cette fonction permet de la fonctionnalité de seam carving horizontalement sur une image 
'''
def apply_horizontal_carving(image1, image2):
    cost_matrix = calculate_cost_matrix(image2)
    seam = sema_detection(cost_matrix)
    image1 = delete_seam(image1, seam)
    image2 = delete_seam(image2, seam)

    return (image1, image2)


'''
Cette fonction permet de la fonctionnalité de seam carving verticalement sur une image 
'''
def apply_vertical_carving(image1, image2):
    image1 = image1.rotate(-90, expand=True)
    image2 = image2.rotate(-90, expand=True)

    image1, image2 = apply_horizontal_carving(image1, image2)

    image1 = image1.rotate(90, expand=True)
    image2 = image2.rotate(90, expand=True)

    return (image1, image2)


'''
La fonction principale
'''
def main():
    file_path = str(sys.argv[1])
    horizontal_percentage = int(sys.argv[2])
    vertical_percentage = int(sys.argv[3])

    if(horizontal_percentage < 0):
        print("Le troisème argument ne peut pas être négatif")
        exit()

    if(vertical_percentage < 0):
        print("Le quatrième argument ne peut pas être négatif")
        exit()
    
    if(horizontal_percentage > 100):
        print("Le troisème argument ne peut pas dépasser la taille de l'image")
        exit()

    if(vertical_percentage > 100):
        print("Le quatrième argument ne peut pas dépasser la taille de l'image")
        exit()

    image1 = Image.open(file_path)
    image2 = gradient(image1)
    
    count = (horizontal_percentage/100) * image1.size[0]
    while count > 0:
        image1, image2 = apply_horizontal_carving(image1, image2)
        count -= 1

    count = (vertical_percentage/100) * image1.size[1]
    while count > 0:
        image1, image2 = apply_vertical_carving(image1, image2)
        count -= 1
    
    image1.save(file_path.split('.')[0] + "_" + str(horizontal_percentage) + '_' + str(vertical_percentage) + '.jpg')



'''
Appel à la fonction principale
'''
main()


    

    

