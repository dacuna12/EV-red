"""
ACN-Sim Tutorial: Lesson 2
Developing a Custom Algorithm
by Zachary Lee
Last updated: 03/19/2019
--
In this lesson we will learn how to develop a custom algorithm and run it using ACN-Sim. For this example we will be
writing an Earliest Deadline First Algorithm. This algorithm is already available as part of the SortingAlgorithm in the
algorithms package, so we will compare the results of our implementation with the included one.
"""

# -- Custom Algorithm --------------------------------------------------------------------------------------------------
import adacharge
from acnportal.algorithms import BaseAlgorithm
from adacharge import *
import numpy
import csv
import pandas



# All custom algorithms should inherit from the abstract class BaseAlgorithm. It is the responsibility of all derived
# classes to implement the schedule method. This method takes as an input a list of EVs which are currently connected
# to the system but have not yet finished charging. Its output is a dictionary which maps a station_id to a list of
# charging rates. Each charging rate is valid for one period measured relative to the current period.
# For Example:
#   * schedule['abc'][0] is the charging rate for station 'abc' during the current period
#   * schedule['abc'][1] is the charging rate for the next period
#   * and so on.
#
# If an algorithm only produces charging rates for the current time period, the length of each list should be 1.
# If this is the case, make sure to also set the maximum resolve period to be 1 period so that the algorithm will be
# called each period. An alternative is to repeat the charging rate a number of times equal to the max recompute period.

# -----------------------------
# Funciones de utilidad
# -----------------------------
def asa_qc(rates, infrastructure, interface, **kwargs):
    # Esta es la función que utilizan en el paper u=u_qc+10-12Ues
    u_qc = adacharge.quick_charge(rates, infrastructure, interface, **kwargs)
    u_es = adacharge.equal_share(rates, infrastructure, interface, **kwargs)
    return u_qc+10e-12*u_es

def rates_unbal(rates, infrastructure, interface, **kwargs):
    # Esta es la función que utilizan en el paper u=u_qc+10-12Ues
    alpha_unbal = 1/100
    # Es un ponderadador de desbalance
    #u_qc = adacharge.quick_charge(rates, infrastructure, interface, **kwargs)
    u_rates = cp.sum(rates)
    u_unbal = rates_phase(rates,infrastructure, interface,**kwargs)
    return u_rates+alpha_unbal*u_unbal

def total_energy(rates, infrastructure, interface, **kwargs):
    return cp.sum(get_period_energy(rates, infrastructure, interface.period))

def rates_phase(rates,infrastructure, interface,**kwargs):
    # Procedimiento que suma las corrientes de la misma fase

    # N: cantidad de vehiculos // T: cantidad de pasos de optimización
    N = rates.shape[0]
    T = rates.shape[1]

    costo_por_paso =[]
    # Se itera en los rates y la fase se levanta del infrasestructure
    for t in range(0,T):
        rates_ab = []
        rates_bc = []
        rates_ca = []
        for i in range(0,N):
            if infrastructure.phases[i] == 30:
                rates_ab.append(rates[i,t])
            elif infrastructure.phases[i] == 150:
                rates_bc.append(rates[i,t])
            else:
                rates_ca.append(rates[i,t])

        # Armo el vector de variables de rates
        g = cp.vstack([cp.sum(rates_ab), cp.sum(rates_bc), cp.sum(rates_ca)])
        # Matriz de transformacion
        M = np.array([[1,-1/2,-1/2],
                      [-1/2,1,-1/2],
                      [-1/2,-1/2,1]])
        # Computo el costo por paso
        costo_por_paso.append(cp.quad_form(g, M))

    #costo = cp.sum(costo_por_paso,axis=0)
    return -cp.sum(costo_por_paso,axis=0)
# -----------------------------

# -----------------------------
# Algoritmo base de ACN
# -----------------------------
class EarliestDeadlineFirstAlgo(BaseAlgorithm):
    """ Algorithm which assigns charging rates to each EV in order or departure time.
    Implements abstract class BaseAlgorithm.
    For this algorithm EVs will first be sorted by departure time. We will then allocate as much current as possible
    to each EV in order until the EV is finished charging or an infrastructure limit is met.
    Args:
        increment (number): Minimum increment of charging rate. Default: 1.
    """

    def __init__(self, increment=1):
        super().__init__()
        self._increment = increment
        self.max_recompute = 1

    def schedule(self, active_evs):
        """ Schedule EVs by first sorting them by departure time, then allocating them their maximum feasible rate.
        Implements abstract method schedule from BaseAlgorithm.
        See class documentation for description of the algorithm.
        Args:
            active_evs (List[EV]): see BaseAlgorithm
        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        # First we define a schedule, this will be the output of our function
        schedule = {ev.station_id: [0] for ev in active_evs}

        # Next, we sort the active_evs by their estimated departure time.
        sorted_evs = sorted(active_evs, key=lambda x: x.estimated_departure)

        # We now iterate over the sorted list of EVs.
        for ev in sorted_evs:
            # First try to charge the EV at its maximum rate. Remember that each schedule value must be a list, even
            #   if it only has one element.
            schedule[ev.station_id] = [self.interface.max_pilot_signal(ev.station_id)]

            # If this is not feasible, we will reduce the rate.
            #   interface.is_feasible() is one way to interact with the constraint set of the network. We will explore
            #   another more direct method in lesson 3.
            while not self.interface.is_feasible(schedule):
                # Since the maximum rate was not feasible, we should try a lower rate.
                schedule[ev.station_id][0] -= self._increment

                # EVs should never charge below 0 (i.e. discharge) so we will clip the value at 0.
                if schedule[ev.station_id][0] < 0:
                    schedule[ev.station_id] = [0]
                    break
        #print(schedule)
        return schedule

# Obtener los datos de los autos
def datos_ev(token, site, start, end, period):
    client = acnsim.acndata_events.DataClient(token)
    docs = client.get_sessions_by_time(site, start, end)
    evs = []
    offset = acnsim.acndata_events._datetime_to_timestamp(start, period)
    for d in docs:
        arrival = acnsim.acndata_events._datetime_to_timestamp(d["connectionTime"], period) - offset
        departure = acnsim.acndata_events._datetime_to_timestamp(d["disconnectTime"], period) - offset
        evs.append([d["sessionID"],d["spaceID"],arrival,departure,d['kWhDelivered']])
    return evs

# Visualizador
def ExportarSimulacion(sim, sim_label):
    # Este procedimiento es el que va a exportar el schedule para luego simularlo en SINCAL

    #-----------------
    # SINCAL
    #-----------------
    # Se cargan los datos del Data Panda Frame
    df = sim.charging_rates_as_df()

    # Renombro los ejes para el SINCAL
    ejes = []
    for i in range(0,len(df.axes[1])):
        ejes.append("P_"+df.axes[1][i])
    df.set_axis(ejes, axis='columns',inplace='True')

    # Los paso a potencia en MW
    P = df*voltage/1e6

    # Exportacion en Excel
    P.to_excel(r'{}\Sincal\{}-{}_SincalProfiles_{}.xlsx'.format(ruta,t_end,t_start,sim_label))

    #----------------------
    # Trafo
    #----------------------
    # Corrientes
    I = corriente_trafo(sim)
    df = pandas.DataFrame(I.transpose())
    df.set_axis(('TR_A','TR_B','TR_C'), axis='columns', inplace='True')
    df.to_excel(r'{}\Corrientes\{}-{}_Isim_{}.xlsx'.format(ruta,t_end,t_start,sim_label))
    # Desbalance
    Unbal = desbalance_trafo(sim)
    df = pandas.DataFrame(Unbal.transpose())
    #df.set_axis('NEMA', axis=0, inplace='True')
    df.to_excel(r'{}\Corrientes\{}-{}_Unbalsim_{}.xlsx'.format(ruta,t_end,t_start,sim_label))
# -----------------------------

# -----------------------------------------------------------------
# Funciones adicionales para el armado de las simulaciones
# -----------------------------------------------------------------
def ArmarSchedule(key):
    tasks = {}

    def task(task_fn):
        tasks[task_fn.__name__] = task_fn

    @task
    def ExecQuickCharge():
        # Agenda
        agendados.append(
            adacharge.AdaptiveChargingAlgorithmOffline(
                [adacharge.ObjectiveComponent(adacharge.quick_charge)], solver=cp.MOSEK, enforce_energy_equality=False
            )
        )
        agendados[-1].register_events(events)
        # Simulacion
        simulaciones.append(acnsim.Simulator(deepcopy(cn), agendados[-1], deepcopy(events), start, period=period))
        agendados[-1].solve()
        simulaciones[-1].run()
    @task
    def ExecAsaQc():
        # Agenda
        agendados.append(
            adacharge.AdaptiveChargingAlgorithmOffline(
                [adacharge.ObjectiveComponent(asa_qc)], solver=cp.MOSEK, enforce_energy_equality=False
            )
        )
        agendados[-1].register_events(events)
        # Simulacion
        simulaciones.append(acnsim.Simulator(deepcopy(cn), agendados[-1], deepcopy(events), start, period=period))
        agendados[-1].solve()
        simulaciones[-1].run()
    @task
    def ExecUncontrolledCharging():
        # Agenda
        agendados.append(algorithms.UncontrolledCharging)
        # Simulacion
        simulaciones.append(acnsim.Simulator(deepcopy(cn), agendados[-1], deepcopy(events), start, period=period))
        simulaciones[-1].run()
    @task
    def ExecTotalEnergy():
        # Agenda
        agendados.append(
            adacharge.AdaptiveChargingAlgorithmOffline(
                [adacharge.ObjectiveComponent(adacharge.total_energy)], solver=cp.MOSEK, enforce_energy_equality=False
            )
        )
        agendados[-1].register_events(events)
        # Simulacion
        simulaciones.append(acnsim.Simulator(deepcopy(cn), agendados[-1], deepcopy(events), start, period=period))
        agendados[-1].solve()
        simulaciones[-1].run()
    @task
    def ExecUnbalMin1():
        # Esta funcion de utilidad que minimiza el desbalance
        agendados.append(
            adacharge.AdaptiveChargingAlgorithmOffline(
                [adacharge.ObjectiveComponent(rates_phase)], solver=cp.MOSEK, enforce_energy_equality=False
            )
        )
        agendados[-1].register_events(events)
        # Simulacion
        simulaciones.append(acnsim.Simulator(deepcopy(cn), agendados[-1], deepcopy(events), start, period=period))
        agendados[-1].solve()
        simulaciones[-1].run()

    @task
    def ExecRatesUnbal():
        # Esta funcion de utilidad que minimiza el desbalance
        agendados.append(
            adacharge.AdaptiveChargingAlgorithmOffline(
                [adacharge.ObjectiveComponent(rates_unbal)], solver=cp.MOSEK, enforce_energy_equality=False
            )
        )
        agendados[-1].register_events(events)
        # Simulacion
        simulaciones.append(acnsim.Simulator(deepcopy(cn), agendados[-1], deepcopy(events), start, period=period))
        agendados[-1].solve()
        simulaciones[-1].run()

    tasks['Exec' + key]()

def corriente_trafo(sim):
    """ Funcion que calcula la corriente por el transformador de la simulacion brindada como argumento

    Args:
        sim (Simulator): simulacion

    Returns:
        Un np.array con cada timestamp en una columna y cada una de las fases de la corriente
    """
    phase_ids = ("Secondary A","Secondary B", "Secondary C")
    currents_dict = acnsim.constraint_currents(sim, constraint_ids=phase_ids)
    currents = np.vstack([currents_dict[phase] for phase in phase_ids])
    currents.transpose()
    return currents

def desbalance_trafo(sim):
    """ Funcion que calcula el desbalance en bornes del transformador

    Args:
        sim (Simulator): simulacion

    Returns:
        Un np.array con los timestamp y el desbalance
    """
    # Variables locales de llamada
    phase_ids = ("Primary A","Primary B", "Primary C")
    return acnsim.current_unbalance(sim,unbalance_type="componentes_inversa",phase_ids=phase_ids)

def energias_por_EV(simulaciones):
    # Obtengo las matrices con las energias - Son PANDAS que tienen en Energias_sim[i_ener].axes[1][i] el cargador
    # Los resultados se almaceana en un numpy_array donde:
    # - filas para cada uno de los autos que entraron al Garage
    # - columnas, en la primera la energia demandada y en las restantes todas las simulaciones

    # Inicializacion de la salida de la funcion
    Resultados_E = np.zeros((len(autos),len(simulaciones)+1))
    #Resultados_E[:,0]=np.array(autos[:][-1])

    # Armo la lista con las simulaciones
    Energias_sim=[]
    for i_ener in range(0, len(simulaciones)):
        Energias_sim.append(obtener_energia_simulada(simulaciones[i_ener], 'porCargador'))

    # Itero en cada auto para obtener la energia simulada
    for i_auto in range(0, len(autos)):
        # Levanto los datos relevantes del auto (cargador al cual esta conectado y en los tiempos que estuvo)
        cargador = autos[i_auto][1]
        t_plug = autos[i_auto][2]
        t_unplug = autos[i_auto][3]
        Resultados_E[i_auto, 0]=autos[i_auto][-1]
        for i_ener in range(0, len(Energias_sim)):
            # Levanto el vector de energias acumuladas
            #E_t = Energias_sim[i_ener][cargador]
            Resultados_E[i_auto, i_ener+1]=Energias_sim[i_ener][cargador][t_unplug]-Energias_sim[i_ener][cargador][t_plug]

    return Resultados_E

# -----------------------------------------------------------------

# Graficas

def graficar_simulaciones(simulaciones,tipo_grafico):
    # Este procedimiento grafica las corrientes simuladas y el desbalance
    # La entrada tipo_grafico es para decidir si usar un subplot o un gráfico agrupado

    ## Variables generales del plot
    guardar_graficas = False
    etiquetas = ["Ia","Ib","Ic"]
    locator = mdates.AutoDateLocator(maxticks=6)
    formatter = mdates.ConciseDateFormatter(locator)

    ## Creo la grafica en funcion del tipo de grafico
    if tipo_grafico == 'unico':
        fig_I, axsI = plt.subplots()
    elif tipo_grafico == 'porFase':
        fig_I, axsI = plt.subplots(len(simulaciones), 1, sharey=True, sharex=True)
    else:
        print(f"El metodo {tipo_grafico} no esta implementado. Se realiza un grafico unico")
        fig_I, axsI = plt.subplots()

    for i_sim in range(0, len(simulaciones)):
        # Levanto las corrientes por fase y el tiempo de cada simulacion
        I = corriente_trafo(simulaciones[i_sim])
        t = mdates.date2num(acnsim.datetimes_array(simulaciones[i_sim]))

        # Se distingue el tipo de ploteo
        if tipo_grafico == 'unico':
            for I_arr, etiqueta in zip(I, etiquetas):
                axsI.plot(t, I_arr, label=etiqueta)
            axsI.set_title(metodos[i_sim])
            # Grilla y referencias
            #plt.grid(True)
            #plt.legend()
        elif tipo_grafico=='porFase':
            for I_arr, etiqueta in zip(I,etiquetas):
                axsI[i_sim].plot(t,I_arr, label=etiqueta)
            axsI[i_sim].set_title(metodos[i_sim])
            # Grilla y referencias

    # Se ajustan los ejes de las graficas
    for ax in axsI:
        ax.set_ylabel("Current (A)")
        for label in ax.get_xticklabels():
            label.set_rotation(40)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.legend()

    # Titulo general
    fig_I.suptitle('Corriente en secundario del TR', fontsize=14)
    if guardar_graficas: plt.savefig(r'{}\Graficas\{}-{}_I-TR.png'.format(ruta, t_end, t_start), dpi=300, bbox_inches='tight')

    # -------------------------------------------------------------------------------
    # Desbalance del trafo - En este caso pongo all en el mismo grafico
    # -------------------------------------------------------------------------------
    fig_unbal, axs_unbal = plt.subplots()
    for i_sim in range(0, len(simulaciones)):
        unbal = desbalance_trafo(simulaciones[i_sim])
        axs_unbal.plot(t, unbal, label=metodos[i_sim])

    # Se ajustan los ejes x de las graficas
    axs_unbal.set_ylabel("Indicador gT*M*g")
    for label in axs_unbal.get_xticklabels():
        label.set_rotation(40)
    axs_unbal.xaxis.set_major_locator(locator)
    axs_unbal.xaxis.set_major_formatter(formatter)
    # Agrego el grid y las etiquetas
    plt.grid(True)
    plt.legend()
    fig_unbal.suptitle('Indicador de desbalance', fontsize=14)
    if guardar_graficas: plt.savefig(r'{}\Graficas\{}-{}_Unbal.png'.format(ruta, t_end, t_start), dpi=300, bbox_inches='tight')

    # -------------------------------------------------------------------------------
    # Energia total
    # -------------------------------------------------------------------------------

    # Energía total solicitda por los autos
    E_dem = 0
    for i in range(0,len(autos)):
        E_dem = E_dem + autos[i][-1]
    # Ploteos de energia en el mismo subplot
    fig_E, axs_E = plt.subplots()
    for i_sim in range(0, len(simulaciones)):
        Energia = obtener_energia_simulada(simulaciones[i_sim])
        axs_E.plot(t, Energia, label=metodos[i_sim])
    # Se ajustan los ejes x de las graficas
    axs_E.set_ylabel("Energia acumulada (kWh)")
    for label in axs_E.get_xticklabels():
        label.set_rotation(40)
    axs_E.xaxis.set_major_locator(locator)
    axs_E.xaxis.set_major_formatter(formatter)
    # Agrego el grid y las etiquetas
    plt.axhline(y=E_dem, color='r',linestyle='--',label='E_dem_EVs')
    plt.grid(True)
    plt.legend()
    fig_E.suptitle('Energía acumulada', fontsize=14)
    if guardar_graficas: plt.savefig(r'{}\Graficas\{}-{}_Energia.png'.format(ruta, t_end, t_start), dpi=300, bbox_inches='tight')

    # -------------------------------------------------------------------------------
    # Energía de cada auto
    # -------------------------------------------------------------------------------

    # Se obtiene la matriz de energias y se hace el ploteo
    Resultados_E = energias_por_EV(simulaciones)
    plot_energias_xy(Resultados_E,metodos)

    # Se muestran todas las graficas
    plt.show()

def obtener_energia_simulada(sim,tipo='acumulada'):
    # Funcion que obtiene el vector de energías simuladas para el despacho elegido
    E_t = sim.charging_rates_as_df()*(voltage/1e3)*(period/60) # Energía en kWh
    if tipo == 'acumulada':
        E_t = E_t.sum(axis=1).cumsum(axis=0)
    else:
        E_t = E_t.cumsum(axis=0)
    return E_t


def plot_energias_xy(Energias,labels):
    ## Plotea el grafico de dispersion entre la energía deseada y la despachada
    f, ax = plt.subplots()
    f.suptitle('Energia demandada y despachada por vehículo')
    # Ploteo de la energia
    for j in range(1,Energias.shape[1]):
        ax.scatter(Energias[:,0], Energias[:,j],marker='o', alpha=0.5,label=labels[j-1])
    # Ploteo la linea diagonal
    ax.plot([0, np.amax(Energias[:,0],axis=0)], [0, np.amax(Energias[:,0],axis=0)], ls="--", c=".3")
    # Ajustes cosmeticos
    ax.set_xlabel("Energia demandanda (kWh)")
    ax.set_ylabel("Energia despachada (kWh)")
    plt.grid(True)
    plt.legend()

# -- Run Simulation ----------------------------------------------------------------------------------------------------
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy

from acnportal import acnsim
from acnportal import algorithms

# -- Experiment Parameters ---------------------------------------------------------------------------------------------
timezone = pytz.timezone("America/Los_Angeles")
t_start = [2018,9,5]
t_end = [2018,9,7]
start = timezone.localize(datetime(t_start[0],t_start[1],t_start[2]))
end = timezone.localize(datetime(t_end[0],t_end[1],t_end[2]))
period = 5  # minute
voltage = 208  # volts
default_battery_power = 32 * voltage / 1000  # kW
exportar = True
site = "caltech"
ruta = 'C:/Users/Diego/Documents/Proyecto FSE/Exportacion'

# Lista de los algoritmos que se desean correr, los posibles son
# - UncontrolledCharging (no se porque no esta andando)
# - TotalEnergy
# - QuickCharge
# - AsaQc
# - UnbalMin1

metodos = ("AsaQc","RatesUnbal")
agendados = []
simulaciones = []
energias_demandadas =[]
autos = []

# -- Network -----------------------------------------------------------------------------------------------------------
cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)

# -- Events ------------------------------------------------------------------------------------------------------------
API_KEY = "DEMO_TOKEN"
events = acnsim.acndata_events.generate_events(
    API_KEY, site, start, end, period, voltage, default_battery_power,energias_demandadas,
)
autos = datos_ev(API_KEY, site, start, end, period)
# -- Scheduling Algorithm & Simulation ----------------------------------------------------------------------------------------------

# Armados de schedules en funcion de los metodos
for i in range(0,len(metodos)):
    ArmarSchedule(metodos[i])

# -- Analysis ----------------------------------------------------------------------------------------------------------
# Se realizan graficas de las corrientes desbalanceadas y del indicador de desbalance
# Las entradas son la lista de simulaciones realizadas y un str que refiere a como se graficas los resultados
# de corriente, el mismo puede valer 'porFase' para obtener un subplot con cada grafico o 'unico' en el que se muestra
# todos los resultados agrupados



# -- Exportacion ----------------------------------------------------------------------------------------------------------
if exportar:
    for i in range(0,len(simulaciones)):
        # Exportacion de la simulacion
        ExportarSimulacion(simulaciones[i],metodos[i])
# Graficas
graficar_simulaciones(simulaciones,'porFase')


