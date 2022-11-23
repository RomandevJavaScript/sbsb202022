import numpy

from array import array
from scipy import special, integrate

import environment
from common import integrators



import shutil
from datetime import timedelta

import environment
from common import UNITS, Params, utils
from common.integrators import adams_bashforth_correction, MAX_ADAMS_BASHFORTH_ORDER

import kawano
from . import eps
import os
import sys
import numpy
import pandas
import time
from datetime import timedelta

from common import UNITS, Params, integrators, parallelization, utils

import kawano

def solve_explicit(f, y_0, h, t_i=0., t_f=5):
    y = array('f', [y_0])
    t = t_i

    while t <= t_f:
        y.append(y[-1] + integrators.euler_correction(y=y[-1], t=t, f=f, h=h))
        t += h

    return y


def solve_implicit(A, B, y_0, h, t_i=0., t_f=5):
    y = array('f', [y_0])
    t = t_i

    while t <= t_f:
        y.append(integrators.implicit_euler(y=y[-1], t=t, A=A, B=B, h=h))
        t += h

    return y


def solve_heun(f, y_0, h, t_i=0., t_f=5):
    y = array('f', [y_0])
    t = t_i

    while t <= t_f:
        y.append(y[-1] + integrators.heun_correction(y=y[-1], t=t, f=f, h=h))
        t += h

    return y


def solve_adams_bashforth(f, y_0, h, t_i=0, t_f=5):
    y = array('f', [y_0])
    fs = array('f', [y_0])
    t = t_i

    while t <= t_f:
        fs.append(f(t, y[-1]))
        y.append(y[-1] + integrators.adams_bashforth_correction(fs[-3:], h))
        t += h

    return y


def explicit_euler_test():

    f = lambda t, y: -2.3 * y

    y_unstable = solve_explicit(f, 1., 1., 0., 5.)
    assert all(numpy.diff(numpy.abs(y_unstable)) >= 0),\
        "Explicit Euler method should be unstable here"
    y_stable = solve_explicit(f, 1., 0.5, 0., 5.)
    assert all(numpy.diff(numpy.abs(y_stable)) <= 0),\
        "Explicit Euler method should be stable here"

    assert y_stable[-1] < eps


def implicit_euler_test():

    y_implicit = numpy.array(solve_implicit(0, -2.3, 1, 1, 0., 5.))
    assert all(y_implicit >= 0), "Implicit Euler method be positive"
    assert all(numpy.diff(numpy.abs(y_implicit)) <= 0), "Implicit Euler method should be stable"

    y_implicit_coarse = numpy.array(solve_implicit(0, -2.3, 1, 2, 0., 5.))
    print(y_implicit_coarse)
    assert all(y_implicit_coarse >= 0), "Implicit Euler method solution be positive"
    assert all(numpy.diff(numpy.abs(y_implicit_coarse)) <= 0), \
        "Implicit Euler method should be stable"


def heun_test():

    f = lambda t, y: -2.3 * y

    exact = numpy.vectorize(lambda t: numpy.exp(-2.3 * t))

    y_euler = solve_explicit(f, 1., 0.5, 0., 5.)
    assert not all(numpy.array(y_euler) >= 0), "Explicit Euler method should be unstable here"
    y_heun = solve_heun(f, 1., 0.5, 0., 5.)
    assert all(numpy.diff(numpy.abs(y_heun)) <= 0), "Heun method should be stable here"

    y_heun_detailed = solve_heun(f, 1., 0.05, 0., 5.)
    assert all((y_heun[1:-1] - exact(numpy.arange(0, 5, 0.5)))[2:] >=
               (y_heun_detailed[1:-1] - exact(numpy.arange(0, 5, 0.05)))[::10][2:]), \
        "Heun method should be more accurate"


# def adams_bashforth_test():

#     f = lambda t, y: -15 * y

#     exact = numpy.vectorize(lambda t: numpy.exp(-15 * t))

#     y_euler = solve_explicit(f, 1., 1./32., 0., 5.)
#     # assert not all(numpy.array(y_euler) >= 0), "Explicit Euler method should be unstable here"
#     # y_heun = solve_adams_bashforth(f, 1., 0.5, 0., 5.)
#     # assert all(numpy.diff(numpy.abs(y_heun)) <= 0), "Heun method should be stable here"

#     y_adams_bashforth = solve_adams_bashforth(f, 1., 1./32., 0., 5.)
#     print(y_euler)
#     print(y_adams_bashforth)
#     assert all((y_euler[1:-1] - exact(numpy.arange(0, 5, 1./32.)))[2:] >=
#                (y_adams_bashforth[1:-1] - exact(numpy.arange(0, 5, 1./32.)))[2:]), \
#         "Heun method should be more accurate"



def gaussian_test():
    func = lambda z: special.jn(3, z)
    adaptive_result, error = integrate.quad(func, 0, 10)
    fixed_result, _ = integrate.fixed_quad(func, 0, 10, n=environment.get('GAUSS_LEGENDRE_ORDER'))
    own_result = integrators.gaussian(func, 0, 10)

    print(adaptive_result, error)
    print(fixed_result)
    print(own_result)

    assert numpy.isclose(integrators.gaussian(func, 0, 10), -integrators.gaussian(func, 10, 0))

    assert adaptive_result - fixed_result < error, "Gauss-Legendre quadrature order is insufficient"
    assert adaptive_result - own_result < error, "Own integrator is inaccurate"


def double_gaussian_test():
    def func(x, y):
        return special.jn(x, y)
    adaptive_result, error = integrate.dblquad(func, 0, 20, lambda x: 1, lambda y: 20)
    own_result = integrators.double_gaussian(func, 0, 20, lambda x: 1, lambda y: 20)

    print(adaptive_result, error)
    print(own_result)

    assert adaptive_result - own_result < error, "Own integrator is inaccurate"


def neutron_lifetime_test():
    from kawano import q as Q, m_e
    q = Q / m_e

    def func(e):
        return e * (e - q)**2 * numpy.sqrt(e**2 - 1)

    adaptive_result, error = integrate.quad(func, 1, q)
    fixed_result, _ = integrate.fixed_quad(func, 1, q, n=environment.get('GAUSS_LEGENDRE_ORDER'))
    own_result = integrators.gaussian(func, 1, q)

    print(adaptive_result, error)
    print(fixed_result)
    print(own_result)

    assert adaptive_result - fixed_result < error, "Gauss-Legendre quadrature order is insufficient"
    assert adaptive_result - own_result < error, "Own integrator is inaccurate"


class Universe(object):
  """ ## Universe
      The master object that governs the calculation. """

  # System state is rendered to the log file each `log_freq` steps
  log_freq = 1
  clock_start = None

  particles = None
  interactions = None

  kawano = None
  kawano_log = None

  oscillations = None

  step_monitor = None

  data = pandas.DataFrame(columns=('aT', 'T', 'a', 'x', 't', 'rho', 'fraction'))

  def __init__(self, folder='logs', plotting=True, params=None, grid=None):
    """
    :param particles: Set of `particle.Particle` to model
    :param interactions: Set of `interaction.Interaction` - quantum interactions \
                         between particle species
    :param folder: Log file path (current `datetime` by default)
    """

    self.particles = []
    self.interactions = []

    self.clock_start = time.time()

    self.params = Params() if not params else params

    # self.graphics = None
    # if utils.getboolenv("PLOT", plotting):
    #     from plotting import Plotting
    #     self.graphics = Plotting()

    self.init_log(folder=folder)

    # Controls parallelization of the collision integrals calculations
    self.PARALLELIZE = utils.getboolenv("PARALLELIZE", True)
    if self.PARALLELIZE:
      parallelization.init_pool()

    self.fraction = 0

    self.step = 1

  def init_kawano(self, datafile='s4.dat', **kwargs):
    kawano.init_kawano(**kwargs)
    self.kawano_log = open(os.path.join(self.folder, datafile), 'w')
    self.kawano_log.write("\t".join(kawano.heading) + "\n")
    self.kawano = kawano
    self.kawano_data = pandas.DataFrame(columns=self.kawano.heading)

  def init_oscillations(self, pattern, particles):
    self.oscillations = (pattern, particles)

  def evolve(self, T_final, export=True):
    """
    ## Main computing routine

    Modeling is carried in steps of scale factor, starting from the initial conditions defined\
    by a single parameter: the initial temperature.

    Initial temperature (e.g. 10 MeV) corresponds to a point in the history of the Universe \
    long before then BBN. Then most particle species are in the thermodynamical equilibrium.

    """

    for particle in self.particles:
      print
      particle

    for interaction in self.interactions:
      print
      interaction

    if self.params.rho is None:
      self.update_particles()
      self.params.update(self.total_energy_density())
    self.save_params()

    while self.params.T > T_final:
      try:
        self.log()
        self.make_step()
        self.save()
        self.step += 1
        self.data.to_pickle(os.path.join(self.folder, "evolution.pickle"))
      except KeyboardInterrupt:
        print
        "Keyboard interrupt!"
        break

    self.log()
    if export:
      self.export()

    return self.data

  def export(self):
    for particle in self.particles:
      print
      particle

    # if self.graphics:
    #     self.graphics.save(self.logfile)

    if self.kawano:
      # if self.graphics:
      #     self.kawano.plot(self.kawano_data, save=self.kawano_log.name)

      self.kawano_log.close()
      print
      kawano.run(self.folder)

      self.kawano_data.to_pickle(os.path.join(self.folder, "kawano.pickle"))

    print
    "Data saved to file {}".format(self.logfile)

    self.data.to_pickle(os.path.join(self.folder, "evolution.pickle"))

  def make_step(self):
    self.integrand(self.params.x, self.params.aT)

    order = min(self.step + 1, 5)
    fs = self.data['fraction'].tail(order - 1).values.tolist()
    fs.append(self.fraction)

    self.params.aT += \
      integrators.adams_bashforth_correction(fs=fs, h=self.params.dy, order=order)
    self.params.x += self.params.dx

    self.params.update(self.total_energy_density())
    if self.step_monitor:
      self.step_monitor(self)

  def add_particles(self, particles):
    for particle in particles:
      particle.set_params(self.params)

    self.particles += particles

  def update_particles(self):
    """ ### 1. Update particles state
        Update particle species distribution functions, check for regime switching,\
        update precalculated variables like energy density and pressure. """
    for particle in self.particles:
      particle.update()

  def init_interactions(self):
    """ ### 2. Initialize non-equilibrium interactions
        Non-equilibrium interactions of different particle species are treated by a\
        numerical integration of the Boltzmann equation for distribution functions in\
        the expanding space-time.

        Depending on the regime of the particle species involved and cosmological parameters, \
        each `Interaction` object populates `Particle.collision_integrals` array with \
        currently active `Integral` objects.
    """
    for interaction in self.interactions:
      interaction.initialize()

  def calculate_collisions(self):
    """ ### 3. Calculate collision integrals """

    particles = [particle for particle in self.particles if particle.collision_integrals]

    with utils.printoptions(precision=2):
      if self.PARALLELIZE:
        for particle in particles:
          parallelization.orders = [
            (particle,
             parallelization.poolmap(particle, 'calculate_collision_integral',
                                     particle.grid.TEMPLATE))
          ]
          for particle, result in parallelization.orders:
            with utils.benchmark(lambda: "I(" + particle.symbol + ") = "
                                         + repr(particle.collision_integral)):
              particle.collision_integral = numpy.array(result.get(1000))
      else:
        for particle in particles:
          with utils.benchmark(lambda: "I(" + particle.symbol + ") = "
                                       + repr(particle.collision_integral)):
            particle.collision_integral = particle.integrate_collisions()

  def update_distributions(self):
    """ ### 4. Update particles distributions """

    if self.oscillations:
      pattern, particles = self.oscillations

      integrals = {A.flavour: A.collision_integral for A in particles}

      for A in particles:
        A.collision_integral = sum(pattern[(A.flavour, B.flavour)] * integrals[B.flavour]
                                   for B in particles)

    for particle in self.particles:
      particle.update_distribution()

  def calculate_temperature_terms(self):
    """ ### 5. Calculate temperature equation terms """

    numerator = 0
    denominator = 0

    for particle in self.particles:
      numerator += particle.numerator
      denominator += particle.denominator

    return numerator, denominator

  def integrand(self, t, y):
    """ ## Temperature equation integrand

        Master equation for the temperature looks like

        \begin{equation}
            \frac{d (aT)}{dx} = \frac{\sum_i{N_i}}{\sum_i{D_i}}
        \end{equation}

        Where $N_i$ and $D_i$ represent contributions from different particle species.

        See definitions for different regimes:
          * [[Radiation|particles/RadiationParticle.py#master-equation-terms]]
          * [[Intermediate|particles/IntermediateParticle.py#master-equation-terms]]
          * [[Dust|particles/DustParticle.py#master-equation-terms]]
          * [[Non-equilibrium|particles/NonEqParticle.py#master-equation-terms]]
    """

    # 1\. Update particles states
    self.update_particles()
    # 2\. Initialize non-equilibrium interactions
    self.init_interactions()
    # 3\. Calculate collision integrals
    self.calculate_collisions()
    # 4\. Update particles distributions
    self.update_distributions()
    # 5\. Calculate temperature equation terms
    numerator, denominator = self.calculate_temperature_terms()
    self.fraction = self.params.x * numerator / denominator

    return self.fraction

  def save_params(self):
    self.data = self.data.append({
      'aT': self.params.aT,
      'T': self.params.T,
      'a': self.params.a,
      'x': self.params.x,
      'rho': self.params.rho,
      'N_eff': self.params.N_eff,
      't': self.params.t,
      'fraction': self.fraction
    }, ignore_index=True)

  def save(self):
    """ Save current Universe parameters into the data arrays or output files """
    self.save_params()

    if self.kawano and self.params.T <= self.kawano.T_kawano:
      #     t[s]         x    Tg[10^9K]   dTg/dt[10^9K/s] rho_tot[g cm^-3]     H[s^-1]
      # n nue->p e  p e->n nue  n->p e nue  p e nue->n  n e->p nue  p nue->n e

      rates = self.kawano.baryonic_rates(self.params.a)

      row = {
        self.kawano.heading[0]: self.params.t / UNITS.s,
        self.kawano.heading[1]: self.params.x / UNITS.MeV,
        self.kawano.heading[2]: self.params.T / UNITS.K9,
        self.kawano.heading[3]: (self.params.T - self.data['T'].iloc[-2])
                                / (self.params.t - self.data['t'].iloc[-2]) * UNITS.s / UNITS.K9,
        self.kawano.heading[4]: self.params.rho / UNITS.g_cm3,
        self.kawano.heading[5]: self.params.H * UNITS.s
      }

      row.update({self.kawano.heading[i]: rate / UNITS.MeV ** 5
                  for i, rate in enumerate(rates, 6)})

      self.kawano_data = self.kawano_data.append(row, ignore_index=True)
      log_entry = "\t".join("{:e}".format(item) for item in self.kawano_data.iloc[-1])

      print
      "KAWANO", log_entry
      self.kawano_log.write(log_entry + "\n")

  def init_log(self, folder=''):
    self.folder = folder
    self.logfile = utils.ensure_path(os.path.join(self.folder, 'log.txt'))
    sys.stdout = utils.Logger(self.logfile)

  def log(self):
    """ Runtime log output """

    # Print parameters every now and then
    if self.step % self.log_freq == 0:
      print('[{clock}] #{step}\tt = {t:e}\taT = {aT:e}\tT = {T:e}\ta = {a:e}\tdx = {dx:e}'
            .format(clock=str(timedelta(seconds=int(time.time() - self.clock_start))),
                    step=self.step,
                    t=self.params.t / UNITS.s,
                    aT=self.params.aT / UNITS.MeV,
                    T=self.params.T / UNITS.MeV,
                    a=self.params.a,
                    dx=self.params.dx / UNITS.MeV))

      # if self.graphics:
      #     self.graphics.plot(self.data)

  def total_energy_density(self):
    return sum(particle.energy_density for particle in self.particles)

  class Universe(object):

    """ ## Universe
        The master object that governs the calculation. """

    # System state is rendered to the evolution.txt file each `export_freq` steps
    export_freq = 100
    log_throttler = None
    clock_start = None

    particles = None
    interactions = None

    kawano = None
    kawano_log = None

    oscillations = None

    step_monitor = None

    data = utils.DynamicRecArray([
      ['aT', 'MeV', UNITS.MeV],
      ['T', 'MeV', UNITS.MeV],
      ['a', None, 1],
      ['x', 'MeV', UNITS.MeV],
      ['t', 's', UNITS.s],
      ['rho', 'MeV^4', UNITS.MeV ** 4],
      ['N_eff', None, 1],
      ['fraction', None, 1],
      ['S', 'MeV^3', UNITS.MeV ** 3]
    ])

    def __init__(self, folder=None, params=None, max_log_rate=2):
      """
      :param folder: Log file path (current `datetime` by default)
      """

      self.particles = []
      self.interactions = []

      self.clock_start = time.time()
      self.log_throttler = utils.Throttler(max_log_rate)

      self.params = params
      if not self.params:
        self.params = Params()

      self.folder = folder
      if self.folder:
        if os.path.exists(folder):
          shutil.rmtree(folder)
        self.init_log(folder=folder)

      self.fraction = 0

      self.step = 1

    def init_kawano(self, datafile='s4.dat', **kwargs):
      kawano.init_kawano(**kwargs)
      if self.folder:
        self.kawano_log = open(os.path.join(self.folder, datafile), 'w')
        self.kawano_log.write("\t".join([col[0] for col in kawano.heading]) + "\n")
      self.kawano = kawano
      self.kawano_data = utils.DynamicRecArray(self.kawano.heading)

    def init_oscillations(self, pattern, particles):
      self.oscillations = (pattern, particles)

    def evolve(self, T_final, export=True, init_time=True):
      """
      ## Main computing routine

      Modeling is carried in steps of scale factor, starting from the initial conditions defined\
      by a single parameter: the initial temperature.

      Initial temperature (e.g. 10 MeV) corresponds to a point in the history of the Universe \
      long before then BBN. Then most particle species are in the thermodynamical equilibrium.

      """
      T_initial = self.params.T

      print("\n\n" + "#" * 32 + " Initial states " + "#" * 32 + "\n")
      for particle in self.particles:
        print(particle)
      print("\n\n" + "#" * 34 + " Log output " + "#" * 34 + "\n")

      for interaction in self.interactions:
        print(interaction)
      print("\n")

      # TODO: test if changing updating particles beforehand changes the computed time
      if init_time:
        self.params.init_time(self.total_energy_density())

      if self.params.rho is None:
        #            self.update_particles()
        self.params.update(self.total_energy_density(), self.total_entropy())
      self.save_params()

      while self.params.T > T_final:
        try:
          self.log()
          self.make_step()
          self.save()
          self.step += 1
          if self.folder and self.step % self.export_freq == 0:
            with open(os.path.join(self.folder, "evolution.txt"), "wb") as f:
              self.data.savetxt(f)
        except KeyboardInterrupt:
          print("\nKeyboard interrupt!")
          sys.exit(1)
          break

      if not (T_initial > self.params.T > 0):
        print("\n(T < 0) or (T > T_initial): suspect numerical instability")
        sys.exit(1)

      self.log()
      if export:
        self.export()

      return self.data

    def export(self):
      print("\n\n" + "#" * 33 + " Final states " + "#" * 33 + "\n")
      for particle in self.particles:
        print(particle)
      print("\n")

      if self.folder:
        if self.kawano:
          self.kawano_log.close()
          print(kawano.run(self.folder))

          with open(os.path.join(self.folder, "kawano.txt"), "wb") as f:
            self.kawano_data.savetxt(f)

        print("Execution log saved to file {}".format(self.logfile))

        with open(os.path.join(self.folder, "evolution.txt"), "wb") as f:
          self.data.savetxt(f)

    def make_step(self):
      self.integrand(self.params.x, self.params.aT)

      if self.step_monitor:
        self.step_monitor(self)

      if environment.get('ADAMS_BASHFORTH_TEMPERATURE_CORRECTION'):
        fs = (list(self.data['fraction'][-MAX_ADAMS_BASHFORTH_ORDER:]) + [self.fraction])

        self.params.aT += adams_bashforth_correction(fs=fs, h=self.params.h)
      else:
        self.params.aT += self.fraction * self.params.h

      self.params.x += self.params.dx
      self.params.update(self.total_energy_density(), self.total_entropy())

      self.log_throttler.update()

    def add_particles(self, particles):
      for particle in particles:
        particle.set_params(self.params)

      self.particles += particles

    def update_particles(self):
      """ ### 1. Update particles state
          Update particle species distribution functions, check for regime switching,\
          update precalculated variables like energy density and pressure. """

      for particle in self.particles:
        particle.update()

    def init_interactions(self):
      """ ### 2. Initialize non-equilibrium interactions
          Non-equilibrium interactions of different particle species are treated by a\
          numerical integration of the Boltzmann equation for distribution functions in\
          the expanding space-time.

          Depending on the regime of the particle species involved and cosmological parameters, \
          each `Interaction` object populates `Particle.collision_integrals` array with \
          currently active `Integral` objects.
      """
      for interaction in self.interactions:
        interaction.initialize()

    def calculate_collisions(self):
      """ ### 3. Calculate collision integrals """

      particles = [particle for particle in self.particles if particle.collision_integrals]

      with utils.printoptions(precision=3, linewidth=100):
        for particle in particles:
          with (utils.benchmark(lambda: "δf/f ({}) = {}".format(particle.symbol,
                                                                particle.collision_integral / particle._distribution * self.params.h),
                                self.log_throttler.output)):
            particle.collision_integral = particle.integrate_collisions()

    def update_distributions(self):
      """ ### 4. Update particles distributions """

      if self.oscillations:
        pattern, particles = self.oscillations

        integrals = {A.flavour: A.collision_integral for A in particles}

        for A in particles:
          A.collision_integral = sum(pattern[(A.flavour, B.flavour)] * integrals[B.flavour]
                                     for B in particles)

      for particle in self.particles:
        particle.update_distribution()

    def calculate_temperature_terms(self):
      """ ### 5. Calculate temperature equation terms """

      numerator = 0
      denominator = 0

      for particle in self.particles:
        numerator += particle.numerator()
        denominator += particle.denominator()

      return numerator, denominator

    def integrand(self, t, y):
      """ ## Temperature equation integrand

          Master equation for the temperature looks like

          \begin{equation}
              \frac{d (aT)}{dx} = \frac{\sum_i{N_i}}{\sum_i{D_i}}
          \end{equation}

          Where $N_i$ and $D_i$ represent contributions from different particle species.

          See definitions for different regimes:
            * [[Radiation|particles/RadiationParticle.py#master-equation-terms]]
            * [[Intermediate|particles/IntermediateParticle.py#master-equation-terms]]
            * [[Dust|particles/DustParticle.py#master-equation-terms]]
            * [[Non-equilibrium|particles/NonEqParticle.py#master-equation-terms]]
      """

      # 1\. Update particles states
      self.update_particles()
      # 2\. Initialize non-equilibrium interactions
      self.init_interactions()
      # 3\. Calculate collision integrals
      self.calculate_collisions()
      # 4\. Update particles distributions
      self.update_distributions()
      # 5\. Calculate temperature equation terms
      numerator, denominator = self.calculate_temperature_terms()

      if environment.get('LOGARITHMIC_TIMESTEP'):
        self.fraction = self.params.x * numerator / denominator
      else:
        self.fraction = numerator / denominator

      return self.fraction

    def save_params(self):
      self.data.append({
        'aT': self.params.aT,
        'T': self.params.T,
        'a': self.params.a,
        'x': self.params.x,
        'rho': self.params.rho,
        'N_eff': self.params.N_eff,
        't': self.params.t,
        'fraction': self.fraction,
        'S': self.params.S
      })

    def save(self):
      """ Save current Universe parameters into the data arrays or output files """
      self.save_params()

      if self.kawano and self.params.T <= self.kawano.T_kawano:

        #     t[s]         x    Tg[10^9K]   dTg/dt[10^9K/s] rho_tot[g cm^-3]     H[s^-1]
        # n nue->p e  p e->n nue  n->p e nue  p e nue->n  n e->p nue  p nue->n e

        rates = self.kawano.baryonic_rates(self.params.a)

        row = {
          self.kawano_data.columns[0]: self.params.t,
          self.kawano_data.columns[1]: self.params.x,
          self.kawano_data.columns[2]: self.params.T,
          self.kawano_data.columns[3]:
            (self.data['T'][-1] - self.data['T'][-2])
            / (self.data['t'][-1] - self.data['t'][-2]),
          self.kawano_data.columns[4]: self.params.rho,
          self.kawano_data.columns[5]: self.params.H
        }

        row.update({self.kawano_data.columns[i]: rate
                    for i, rate in enumerate(rates, 6)})

        self.kawano_data.append(row)

        if self.log_throttler.output:
          print("KAWANO", self.kawano_data.row_repr(-1, names=True))
        self.kawano_log.write(self.kawano_data.row_repr(-1) + "\n")

    def init_log(self, folder=''):
      self.logfile = utils.ensure_path(os.path.join(self.folder, 'log.txt'))
      sys.stdout = utils.Logger(self.logfile)

    def log(self):
      """ Runtime log output """

      # Print parameters every now and then
      if self.log_throttler.output:
        print('[{clock}] #{step}\tt = {t:e} s\taT = {aT:e} MeV\tT = {T:e} MeV'
              '\tδaT/aT = {daT:e}\tS = {S:e} MeV^3'
              .format(clock=timedelta(seconds=int(time.time() - self.clock_start)),
                      step=self.step,
                      t=self.params.t / UNITS.s,
                      aT=self.params.aT / UNITS.MeV,
                      T=self.params.T / UNITS.MeV,
                      daT=self.fraction * self.params.h / self.params.aT,
                      S=self.params.S / UNITS.MeV ** 3))

    def total_entropy(self):
      return sum(particle.entropy for particle in self.particles) * self.params.a ** 3

    def total_energy_density(self):
      return sum(particle.energy_density for particle in self.particles)
