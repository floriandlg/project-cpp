from v_numpy import GaborNumpy, PhasorNumpy
from v_taichi import GaborTaichi, PhasorTaichi
from v_python_pur import GaborPur, PhasorPur

import time
import matplotlib.pyplot as plt
import numpy




def compare_one_gabor(RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
    out = []
    out.append([RESOLUTION,FREQUENCY,ORIENTATION,NB_CELLS])
    
    start = time.time()
    GaborNumpy(RESOLUTION,FREQUENCY,ORIENTATION,NB_CELLS).gabor_noise()
    out.append(time.time() - start)

    start = time.time()
    GaborPur(RESOLUTION,FREQUENCY,ORIENTATION,NB_CELLS).gabor_noise()
    out.append(time.time() - start)

    start = time.time()
    GaborTaichi(RESOLUTION,FREQUENCY,ORIENTATION,NB_CELLS).calcul_gabor_noise()
    out.append(time.time() - start)

    print(out)
    return out
        


def compare_one_phasor(RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
    out = []
    out.append([RESOLUTION,FREQUENCY,ORIENTATION,NB_CELLS])
    
    start = time.time()
    PhasorNumpy(RESOLUTION,FREQUENCY,ORIENTATION,NB_CELLS).phasor_noise()
    out.append(time.time() - start)

    start = time.time()
    PhasorPur(RESOLUTION,FREQUENCY,ORIENTATION,NB_CELLS).phasor_noise()
    out.append(time.time() - start)

    start = time.time()
    PhasorTaichi(RESOLUTION,FREQUENCY,ORIENTATION,NB_CELLS).calcul_phasor_noise()
    out.append(time.time() - start)

    print(out)
    return out









def time_over_res(sResolution, fResolution, n, Frequency=75, Orientation=0.5, NB_cells=10):
    """

        Graph en fonction de la resolution, 
        n = nombre de tests entre les deux resolutions
    
    """


    pha = [[],[],[],[]]
    step = int((fResolution - sResolution) / (n - 1))
    for i in range(n):
        bop = compare_one_phasor(sResolution + i * step, Frequency, Orientation, NB_cells)
        pha[0].append(bop[0])
        pha[1].append(bop[1])
        pha[2].append(bop[2])
        pha[3].append(bop[3])

    x = numpy.linspace(sResolution, fResolution, n)
    plt.title('Comparaison en fonction de la résolution')
    plt.plot(x , pha[2], 'o-', label='Python pur',color='C1')
    plt.plot(x , pha[1], 'o-', label='Numpy',color='C0')
    plt.plot(x , pha[3], 'o-', label='Taichi',color='C2')
    plt.legend(loc="upper left")
    plt.xlabel("Racine carré du nombre de pixel")
    plt.ylabel("Temps d'exécution (s)")
    plt.show()


def time_over_kernel(sNB_CELLS, fNB_CELLS, step, Frequency=75, Orientation=0.5, Resolution=700):
    """

        Graph en fonction du nombre de noyaux, 
        step = nombre de tests entre les deux nombre de noyaux
    
    """


    pha = [[],[],[],[]]
    x = []

    while sNB_CELLS <= fNB_CELLS:
        x.append(sNB_CELLS)
        bop = compare_one_phasor(Resolution, Frequency, Orientation, sNB_CELLS)
        sNB_CELLS += step
        pha[0].append(bop[0])
        pha[1].append(bop[1])
        pha[2].append(bop[2])
        pha[3].append(bop[3])

        
    plt.title('Comparaison en fonction du nombre de noyaux')
    plt.plot(x , pha[2], 'o-', label='Python pur',color='C1')
    plt.plot(x , pha[1], 'o-', label='Numpy',color='C0')
    plt.plot(x , pha[3], 'o-', label='Taichi',color='C2')
    plt.legend(loc="upper left")
    plt.xlabel("Racine carré du nombre de noyaux")
    plt.ylabel("Temps d'exécution (s)")
    plt.show()


def time_over_res_fixed_kernel(sResolution, fResolution, n, size_kernel, Frequency=75, Orientation=0.5):
    """

        Graphe en fonction de la resolution avec nombre de noyaux proportionnel,
        size_kernel = taille d'une cellule d'un noyau
        n = nombre de tests entre les deux résolutions

    """
    

    pha = [[],[],[],[]]
    step = int((fResolution - sResolution) / (n - 1))
    for i in range(n):
        bop = compare_one_phasor(sResolution + i * step, Frequency, Orientation, (sResolution + i * step) // size_kernel)
        pha[0].append(bop[0])
        pha[1].append(bop[1])
        pha[2].append(bop[2])
        pha[3].append(bop[3])


    x = numpy.linspace(sResolution, fResolution, n)
    plt.title('Comparaison en fonction de la résolution (nombre de noyaux proportionnel au nombre de pixels)')
    plt.plot(x , pha[2], 'o-', label='Python pur',color='C1')
    plt.plot(x , pha[1], 'o-', label='Numpy',color='C0')
    plt.plot(x , pha[3], 'o-', label='Taichi',color='C2')
    plt.legend(loc="upper left")
    plt.xlabel("Racine carré du nombre de pixel")
    plt.ylabel("Temps d'exécution (s)")
    plt.show()



def time_3d(sResolution, fResolution, n, sNB_CELLS, fNB_CELLS, m, Frequency=75, Orientation=0.5):
    """
    
        Graph en fonction de la résolution et du nombre de noyaux,
        n = nombre de resolutions à tester 
        m = nombre de noyaux à tester 

        nombre de tests = n * m 

    """


    res = numpy.linspace(sResolution, fResolution, n)
    kernel = numpy.linspace(sNB_CELLS, fNB_CELLS, m)

    
    res, kernel = numpy.meshgrid(res, kernel)
    print(res,kernel)

    Z1 = numpy.zeros(res.shape)
    Z2 = numpy.zeros(res.shape)
    Z3 = numpy.zeros(res.shape)

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            aaaaaaaa = compare_one_phasor(int(res[i][j]), Frequency, Orientation, int(kernel[i][j]))
            Z1[i][j] = aaaaaaaa[1]
            Z2[i][j] = aaaaaaaa[2]
            Z3[i][j] = aaaaaaaa[3]
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_wireframe(res, kernel, Z1,color='C0')
    ax.plot_wireframe(res, kernel, Z2,color='C1')
    ax.plot_wireframe(res, kernel, Z3,color='C2')
    ax.legend(['Numpy', 'Python pur','Taichi'])

    plt.show()
    






'''

graphes des exemples :

'''


#time_3d(100, 1000, 10, 3, 30, 10)
#time_over_res_fixed_kernel(100,1200,12,50)
#time_over_kernel(2,30,2)
#time_over_res(100, 1000, 10)
