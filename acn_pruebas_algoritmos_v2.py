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
        g = cp.vstack([cp.sum(rates_ab,axis=0), cp.sum(rates_bc, axis=0), cp.sum(rates_ca, axis=0)])
        # Matriz de transformacion
        M = np.array([[1,-1/2,-1/2],
                      [-1/2,1,-1/2],
                      [-1/2,-1/2,1]])
        # Computo el costo por paso
        costo_por_paso.append(cp.quad_form(g, M))

    costo = cp.sum(costo_por_paso,axis=1)
    return costo
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
    df.set_axis(('TR_C','TR_A','TR_B'), axis='columns', inplace='True')
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
        Un np.array con cada timestamp en una columna y cada una de las fases de la corriente
    """
    # Variables locales de llamada
    phase_ids = ("Primary A","Primary B", "Primary C")
    return acnsim.current_unbalance(sim,unbalance_type="NEMA_ponderado",phase_ids=phase_ids)

# -----------------------------------------------------------------

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
t_end = [2018,9,6]
start = timezone.localize(datetime(t_start[0],t_start[1],t_start[2]))
end = timezone.localize(datetime(t_end[0],t_end[1],t_end[2]))
period = 5  # minute
voltage = 208  # volts
default_battery_power = 32 * voltage / 1000  # kW
site = "caltech"
ruta = 'C:/Users/Diego/Documents/Proyecto FSE/Exportacion'

# Lista de los algoritmos que se desean correr, los posibles son
# - UncontrolledCharging (no se porque no esta andando)
# - TotalEnergy
# - QuickCharge
# - AsaQc
# - UnbalMin1

metodos = ("AsaQc","UnbalMin1")
agendados = []
simulaciones = []

# -- Network -----------------------------------------------------------------------------------------------------------
cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)

# -- Events ------------------------------------------------------------------------------------------------------------
API_KEY = "DEMO_TOKEN"
events = acnsim.acndata_events.generate_events(
    API_KEY, site, start, end, period, voltage, default_battery_power
)

# -- Scheduling Algorithm & Simulation ----------------------------------------------------------------------------------------------

# Armados de schedules en funcion de los metodos
for i in range(0,len(metodos)):
    ArmarSchedule(metodos[i])

# -- Analysis ----------------------------------------------------------------------------------------------------------
# We can now compare the two algorithms side by side by looking that the plots of aggregated current.
# We see from these plots that our implementation matches th included one quite well. If we look closely however, we
# might see a small difference. This is because the included algorithm uses a more efficient bisection based method
# instead of our simpler linear search to find a feasible rate.

# Set locator and formatter for datetimes on x-axis.
locator = mdates.AutoDateLocator(maxticks=6)
formatter = mdates.ConciseDateFormatter(locator)
fig, axs = plt.subplots(1, len(simulaciones), sharey=True, sharex=True)

# Exportacion de la simulacion a EXCEL
for i in range(0,len(simulaciones)):
    # Exportacion de la simulacion
    ExportarSimulacion(simulaciones[i],metodos[i])
    # Ejes
    axs[i].plot(mdates.date2num(acnsim.datetimes_array(simulaciones[i])), acnsim.aggregate_current(simulaciones[i]), label=metodos[i])
    axs[i].set_title(metodos[i])

# Se ajustan los ejes de las graficas
for ax in axs:
    ax.set_ylabel("Current (A)")
    for label in ax.get_xticklabels():
        label.set_rotation(40)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
plt.show()

