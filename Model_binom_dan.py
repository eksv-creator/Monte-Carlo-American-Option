import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# Часть 1: Параметры модели
# ----------------------------------------------------------------------------

print("=" * 80)
print("ЗАДАНИЕ 5: МЕТОД МОНТЕ-КАРЛО ДЛЯ АМЕРИКАНСКОГО PUT-ОПЦИОНА")
print("=" * 80)

# Параметры из задания
S0 = 100  # начальная цена
mu = 0.08  # дрейф в реальном мире
sigma = 0.3  # волатильность
r = 0.05  # безрисковая ставка
T = 1.0  # время до экспирации (год)
K = 105  # страйк-цена

# Параметры для симуляции
N = 50  # количество временных шагов (будем варьировать)
M = 10000  # количество траекторий (будем варьировать)
dt = T / N  # шаг по времени

print(f"Параметры модели:")
print(f"  S0 = {S0}")
print(f"  μ = {mu} (дрейф в реальном мире)")
print(f"  σ = {sigma} (волатильность)")
print(f"  r = {r} (безрисковая ставка)")
print(f"  T = {T} (время до экспирации)")
print(f"  K = {K} (страйк-цена)")
print(f"  N = {N} (количество временных шагов)")
print(f"  M = {M} (количество траекторий)")
print(f"  Δt = {dt:.4f}")


# ----------------------------------------------------------------------------
# Часть 2: Функции для генерации траекторий GBM
# ----------------------------------------------------------------------------

def generate_gbm_trajectories(S0, mu, sigma, T, N, M):
    """Генерация траекторий геометрического броуновского движения"""
    dt = T / N
    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    for i in range(1, N + 1):
        Z = np.random.randn(M)
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return S


def generate_gbm_trajectories_risk_neutral(S0, r, sigma, T, N, M):
    """Генерация траекторий в риск-нейтральной мере"""
    dt = T / N
    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    for i in range(1, N + 1):
        Z = np.random.randn(M)
        S[:, i] = S[:, i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return S


# ----------------------------------------------------------------------------
# Часть 3: Алгоритм Лонгстаффа-Шварца
# ----------------------------------------------------------------------------

def longstaff_schwartz_american_put(S0, K, r, sigma, T, N, M, degree=2):
    """
    Алгоритм Лонгстаффа-Шварца для оценки американского put-опциона
    """
    dt = T / N
    discount_factor = np.exp(-r * dt)

    # Генерация траекторий в риск-нейтральной мере
    S = generate_gbm_trajectories_risk_neutral(S0, r, sigma, T, N, M)

    # Матрица выплат при исполнении
    cash_flow = np.zeros((M, N + 1))
    cash_flow[:, -1] = np.maximum(K - S[:, -1], 0)

    # Матрица оптимальных моментов исполнения
    exercise_time = np.full(M, N)

    # Обратная индукция
    for t in range(N - 1, 0, -1):
        # Текущая цена
        St = S[:, t]

        # Выплата при немедленном исполнении
        immediate_payoff = np.maximum(K - St, 0)

        # Отбираем траектории "в деньгах"
        in_the_money = immediate_payoff > 0

        if np.sum(in_the_money) > 0:
            # Вычисляем дисконтированную будущую стоимость
            future_value = cash_flow[:, t + 1] * discount_factor

            # Регрессия для оценки стоимости продолжения
            X = St[in_the_money].reshape(-1, 1)
            y = future_value[in_the_money]

            # Создаем полиномиальные признаки
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X)

            # Линейная регрессия
            model = LinearRegression()
            model.fit(X_poly, y)

            # Прогноз стоимости продолжения
            continuation_value_pred = model.predict(X_poly)

            # Сравниваем немедленное исполнение и продолжение
            execute_now = immediate_payoff[in_the_money] > continuation_value_pred

            # Обновляем cash flow
            execute_indices = np.where(in_the_money)[0][execute_now]
            cash_flow[execute_indices, t] = immediate_payoff[execute_indices]
            cash_flow[execute_indices, t + 1:] = 0

            # Запоминаем момент исполнения
            exercise_time[execute_indices] = t

    # Вычисляем начальную стоимость опциона
    price = 0
    for t in range(1, N + 1):
        price += np.mean(cash_flow[:, t] * np.exp(-r * t * dt))

    return price, exercise_time, S


# ----------------------------------------------------------------------------
# Часть 4: Исследование сходимости по числу траекторий M
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ИССЛЕДОВАНИЕ СХОДИМОСТИ ПО ЧИСЛУ ТРАЕКТОРИЙ")
print("=" * 80)

M_values = [100, 500, 1000, 2000, 5000, 10000, 20000]
N_fixed = 50
degree = 2

prices_by_M = []
std_errors = []

print(f"\nОценка стоимости для N={N_fixed}, степень полинома={degree}:")
print("-" * 60)
print(f"{'M':>8} {'Цена':>10} {'Стд. ошибка':>12} {'Запусков':>10}")
print("-" * 60)

for M_val in M_values:
    n_runs = 10 if M_val <= 5000 else 5
    run_prices = []

    for run in range(n_runs):
        price, _, _ = longstaff_schwartz_american_put(S0, K, r, sigma, T, N_fixed, M_val, degree)
        run_prices.append(price)

    mean_price = np.mean(run_prices)
    std_error = np.std(run_prices) / np.sqrt(n_runs)

    prices_by_M.append(mean_price)
    std_errors.append(std_error)

    print(f"{M_val:>8} {mean_price:>10.4f} {std_error:>12.4f} {n_runs:>10}")

# Построение графика сходимости по M
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.errorbar(M_values, prices_by_M, yerr=std_errors, fmt='o-',
             capsize=5, capthick=2, linewidth=2)
plt.xlabel('Число траекторий M', fontsize=12)
plt.ylabel('Стоимость опциона', fontsize=12)
plt.title('Сходимость метода Монте-Карло по числу траекторий', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xscale('log')

plt.subplot(1, 2, 2)
expected_std = std_errors[0] * np.sqrt(M_values[0]) / np.sqrt(np.array(M_values))
plt.loglog(M_values, std_errors, 'o-', label='Фактическая', linewidth=2)
plt.loglog(M_values, expected_std, '--', label='Ожидаемая (1/√M)', linewidth=2)
plt.xlabel('log(M)', fontsize=12)
plt.ylabel('log(Стд. ошибка)', fontsize=12)
plt.title('Скорость уменьшения стандартной ошибки', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Часть 5: Исследование влияния числа временных шагов N
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ИССЛЕДОВАНИЕ ВЛИЯНИЯ ЧИСЛА ВРЕМЕННЫХ ШАГОВ N")
print("=" * 80)

N_values = [10, 25, 50, 75, 100, 200]
M_fixed = 10000
degree = 2

prices_by_N = []

print(f"\nОценка стоимости для M={M_fixed}, степень полинома={degree}:")
print("-" * 60)
print(f"{'N':>8} {'Цена':>10} {'Δt':>10} {'Исп. сейчас (%)':>15}")
print("-" * 60)

for N_val in N_values:
    price, exercise_time, S = longstaff_schwartz_american_put(
        S0, K, r, sigma, T, N_val, M_fixed, degree
    )

    early_exercise_pct = np.sum(exercise_time < N_val) / M_fixed * 100

    prices_by_N.append(price)

    print(f"{N_val:>8} {price:>10.4f} {T / N_val:>10.4f} {early_exercise_pct:>15.1f}")

# Построение графиков влияния N
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(N_values, prices_by_N, 'o-', linewidth=2, markersize=8)
plt.xlabel('Число временных шагов N', fontsize=12)
plt.ylabel('Стоимость опциона', fontsize=12)
plt.title('Влияние дискретизации времени на стоимость', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
price_N50, exercise_time_N50, S_N50 = longstaff_schwartz_american_put(
    S0, K, r, sigma, T, 50, 10000, degree
)
plt.hist(exercise_time_N50, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Момент исполнения', fontsize=12)
plt.ylabel('Количество траекторий', fontsize=12)
plt.title('Распределение оптимальных моментов исполнения (N=50)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Часть 6: Сравнение с биномиальной моделью (ИСПРАВЛЕННАЯ)
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("СРАВНЕНИЕ С БИНОМИАЛЬНОЙ МОДЕЛЬЮ")
print("=" * 80)


# Функция для биномиальной модели (исправленная)
def binomial_american_put(S0, K, r, sigma, T, N):
    """
    Биномиальная модель для американского put (Кокс-Росс-Рубинштейн)
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Построение дерева цен
    S = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            S[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Обратная индукция
    V = np.zeros((N + 1, N + 1))
    V[:, N] = np.maximum(K - S[:, N], 0)

    for t in range(N - 1, -1, -1):
        for i in range(t + 1):
            continuation = discount * (q * V[i, t + 1] + (1 - q) * V[i + 1, t + 1])
            exercise = max(K - S[i, t], 0)  # Используем max() из Python
            V[i, t] = max(continuation, exercise)  # Используем max() из Python

    return V[0, 0]


# Сравнение для разных N
print(f"\nСравнение методов для T={T}, K={K}:")
print("-" * 70)
print(f"{'N':>8} {'Биномиальная':>15} {'Монте-Карло':>15} {'Разница':>12} {'Отн. (%)':>10}")
print("-" * 70)

comparison_results = []
for N_val in [10, 20, 30, 50, 100]:
    # Биномиальная модель
    bin_price = binomial_american_put(S0, K, r, sigma, T, N_val)

    # Метод Монте-Карло
    mc_prices = []
    for _ in range(5):
        price, _, _ = longstaff_schwartz_american_put(S0, K, r, sigma, T, N_val, 10000, 2)
        mc_prices.append(price)
    mc_price = np.mean(mc_prices)
    mc_std = np.std(mc_prices)

    diff = mc_price - bin_price
    rel_diff = diff / bin_price * 100

    comparison_results.append((N_val, bin_price, mc_price, mc_std, diff, rel_diff))

    print(f"{N_val:>8} {bin_price:>15.4f} {mc_price:>15.4f} ±{mc_std:.4f} "
          f"{diff:>12.4f} {rel_diff:>10.2f}")

# График сравнения
plt.figure(figsize=(10, 6))
N_vals = [r[0] for r in comparison_results]
bin_prices = [r[1] for r in comparison_results]
mc_prices = [r[2] for r in comparison_results]
mc_stds = [r[3] for r in comparison_results]

plt.plot(N_vals, bin_prices, 'o-', linewidth=2, markersize=8,
         label='Биномиальная модель', color='blue')
plt.plot(N_vals, mc_prices, 's-', linewidth=2, markersize=8,
         label='Монте-Карло (LSM)', color='red')

# Полосы погрешности для Monte Carlo
plt.fill_between(N_vals,
                 np.array(mc_prices) - np.array(mc_stds),
                 np.array(mc_prices) + np.array(mc_stds),
                 alpha=0.2, color='red')

plt.xlabel('Число временных шагов N', fontsize=12)
plt.ylabel('Стоимость опциона', fontsize=12)
plt.title('Сравнение биномиальной модели и метода Монте-Карло', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Часть 7: Анализ влияния базисных функций
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("АНАЛИЗ ВЛИЯНИЯ БАЗИСНЫХ ФУНКЦИЙ")
print("=" * 80)

degrees = [1, 2, 3, 4, 5]
M_test = 10000
N_test = 50

print(f"\nВлияние степени полинома на точность (M={M_test}, N={N_test}):")
print("-" * 70)
print(f"{'Степень':>8} {'Цена':>12} {'Стд. откл.':>12} {'Исп. (%)':>12}")
print("-" * 70)

prices_by_degree = []
for degree in degrees:
    n_runs = 5
    prices = []
    early_exercise_counts = []

    for _ in range(n_runs):
        price, exercise_time, _ = longstaff_schwartz_american_put(
            S0, K, r, sigma, T, N_test, M_test, degree
        )
        prices.append(price)
        early_exercise_counts.append(np.sum(exercise_time < N_test) / M_test * 100)

    mean_price = np.mean(prices)
    std_price = np.std(prices)
    mean_early_exercise = np.mean(early_exercise_counts)

    prices_by_degree.append((degree, mean_price, std_price, mean_early_exercise))

    print(f"{degree:>8} {mean_price:>12.4f} ±{std_price:.4f} "
          f"{mean_early_exercise:>12.1f}")

# График влияния степени полинома
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
degrees_vals = [p[0] for p in prices_by_degree]
prices_vals = [p[1] for p in prices_by_degree]
prices_err = [p[2] for p in prices_by_degree]

plt.errorbar(degrees_vals, prices_vals, yerr=prices_err,
             fmt='o-', capsize=5, linewidth=2)
plt.xlabel('Степень полинома', fontsize=12)
plt.ylabel('Стоимость опциона', fontsize=12)
plt.title('Влияние степени полинома на оценку стоимости', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
single_trajectory = generate_gbm_trajectories_risk_neutral(S0, r, sigma, T, N_test, 1)[0]
payoff = np.maximum(K - single_trajectory, 0)

x = single_trajectory[:-1].reshape(-1, 1)
y = payoff[:-1]

plt.scatter(single_trajectory[:-1], payoff[:-1], alpha=0.6,
            label='Наблюдения', color='gray')

for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X_poly, y)

    x_grid = np.linspace(single_trajectory.min(), single_trajectory.max(), 100).reshape(-1, 1)
    x_grid_poly = poly.transform(x_grid)
    y_pred = model.predict(x_grid_poly)

    plt.plot(x_grid, y_pred, linewidth=2,
             label=f'Полином {degree}-й степени')

plt.xlabel('Цена акции S', fontsize=12)
plt.ylabel('Выплата put-опциона', fontsize=12)
plt.title('Аппроксимация функции выплат полиномами', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Часть 8: Итоговый анализ
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ИТОГОВЫЙ АНАЛИЗ И ВЫВОДЫ")
print("=" * 80)

print(f"\n1. ОЦЕНКА СТОИМОСТИ АМЕРИКАНСКОГО PUT-ОПЦИОНА:")
print(f"   Метод Лонгстаффа-Шварца: {prices_by_M[-1]:.4f} ± {std_errors[-1]:.4f}")
print(f"   (при M={M_values[-1]}, N={N_fixed}, степень полинома={degree})")

print(f"\n2. СХОДИМОСТЬ МЕТОДА:")
print(f"   • Стандартная ошибка уменьшается как O(1/√M)")
print(f"   • При M=10,000 ошибка составляет ~{std_errors[-2]:.4f}")
print(f"   • При M=20,000 ошибка снижается до ~{std_errors[-1]:.4f}")

print(f"\n3. ВЛИЯНИЕ ДИСКРЕТИЗАЦИИ ВРЕМЕНИ:")
print(f"   • При N=10: цена = {prices_by_N[0]:.4f}")
print(f"   • При N=50: цена = {prices_by_N[3]:.4f}")
print(f"   • При N=100: цена = {prices_by_N[-1]:.4f}")

print(f"\n4. СРАВНЕНИЕ С БИНОМИАЛЬНОЙ МОДЕЛЬЮ:")
best_bin_price = comparison_results[-1][1]
best_mc_price = comparison_results[-1][2]
diff = best_mc_price - best_bin_price
print(f"   • Биномиальная модель (N=100): {best_bin_price:.4f}")
print(f"   • Метод Монте-Карло (N=100): {best_mc_price:.4f}")
print(f"   • Разница: {diff:.4f} ({abs(diff) / best_bin_price * 100:.2f}%)")

print(f"\n5. ВЫБОР БАЗИСНЫХ ФУНКЦИЙ:")
print(f"   • Полином 1-й степени: {prices_by_degree[0][1]:.4f}")
print(f"   • Полином 2-й степени: {prices_by_degree[1][1]:.4f}")
print(f"   • Полином 3-й степени: {prices_by_degree[2][1]:.4f}")
print(f"   • Полином 4-й степени: {prices_by_degree[3][1]:.4f}")
print(f"   • Полином 5-й степени: {prices_by_degree[4][1]:.4f}")
print(f"   • Рекомендация: использовать полином 2-й или 3-й степени")

print(f"\n6. ОПТИМАЛЬНАЯ СТРАТЕГИЯ ИСПОЛНЕНИЯ:")
print(f"   • Примерный % досрочного исполнения: {prices_by_degree[1][3]:.1f}%")

print(f"\n7. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
print(f"   • Для точной оценки: M ≥ 10,000, N ≥ 50")
print(f"   • Использовать полиномиальные функции 2-3 степени")
print(f"   • Выполнять 5-10 запусков для оценки стандартной ошибки")

print("\n" + "=" * 80)
print("ЗАДАНИЕ 5 ВЫПОЛНЕНО")
print("=" * 80)