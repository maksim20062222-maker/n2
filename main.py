import numpy as np
import matplotlib.pyplot as plt
import os


class DipolePattern:
    def __init__(self, freq_ghz=10.5, two_l_over_lambda=0.9, data_file="123.txt"):
        self.c = 299_792_458.0
        self.f = freq_ghz * 1e9
        self.two_l_over_lambda = two_l_over_lambda
        self.lmbda = self.c / self.f
        self.l = 0.5 * two_l_over_lambda * self.lmbda
        self.k = 2 * np.pi / self.lmbda
        self.kl = self.k * self.l
        self.data_file = data_file

    def F_theta(self, theta):


        th = np.array(theta, dtype=float)


        num = np.cos(self.kl * np.cos(th)) - np.cos(self.kl)


        den = np.sin(self.kl) * np.sin(th)


        eps = 1e-12
        den_safe = np.where(np.abs(den) < eps, eps, den)


        F = np.abs(num / den_safe)

        return F

    def compute_analytic_D(self, n_theta=3601):

        print("\n" + "=" * 60)
        print("АНАЛИТИЧЕСКИЙ РАСЧЕТ")
        print("=" * 60)
        print(f"Параметры:")
        print(f"  f = {self.f / 1e9} ГГц")
        print(f"  2l/λ = {self.two_l_over_lambda}")
        print(f"  kl = {self.kl:.4f} рад")
        print(f"  sin(kl) = {np.sin(self.kl):.6f}")
        print(f"  cos(kl) = {np.cos(self.kl):.6f}")


        theta = np.linspace(0.0, 2 * np.pi, n_theta)


        theta_mod = np.mod(theta, np.pi)
        theta_mod = np.where(theta_mod == 0, 1e-10, theta_mod)


        F = self.F_theta(theta_mod)


        Fmax = F.max()
        if Fmax == 0:
            raise RuntimeError("Максимум поля равен нулю")
        F_norm = F / Fmax



        theta_int = np.linspace(0.0, np.pi, 1001)
        F_int = self.F_theta(theta_int)
        F_int_norm = F_int / F_int.max()
        integrand = F_int_norm ** 2 * np.sin(theta_int)

        try:
            integral = np.trapezoid(integrand, theta_int)
        except AttributeError:
            integral = np.trapz(integrand, theta_int)

        Dmax = 2.0 / integral


        D_theta = F_norm ** 2 * Dmax

        print(f"\nРезультаты:")
        print(f"  Интеграл = {integral:.6f}")
        print(f"  Dmax = {Dmax:.6f} ({10 * np.log10(Dmax):.3f} дБ)")

        return theta, D_theta, Dmax

    def read_simulation(self):

        print("\n" + "=" * 60)
        print("ЗАГРУЗКА ДАННЫХ ИЗ CST")
        print("=" * 60)
        print(f"Файл: {self.data_file}")

        if not os.path.exists(self.data_file):
            print(f"ОШИБКА: Файл не найден!")
            return np.array([]), np.array([]), np.array([])

        print(f"Размер файла: {os.path.getsize(self.data_file)} байт")

        with open(self.data_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        print(f"Прочитано строк: {len(lines)}")

        thetas = []
        values_db = []


        data_started = False
        for i, line in enumerate(lines):
            line = line.strip()


            if not data_started and ('---' in line or 'Theta' in line):
                data_started = True
                continue

            if not data_started or not line:
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    theta = float(parts[0])

                    value_db = float(parts[2])

                    if 0 <= theta <= 180:
                        thetas.append(theta)
                        values_db.append(value_db)
                except (ValueError, IndexError) as e:
                    continue

        if not thetas:
            print("ОШИБКА: Не удалось прочитать данные!")
            print("Первые 10 строк файла:")
            for i, line in enumerate(lines[:10]):
                print(f"  {i}: {repr(line)}")
            return np.array([]), np.array([]), np.array([])

        theta_deg = np.array(thetas)
        values_db = np.array(values_db)

        print(f"Загружено точек: {len(theta_deg)}")
        print(f"Диапазон углов: {theta_deg.min():.1f}° - {theta_deg.max():.1f}°")
        print(f"Максимум в дБ: {values_db.max():.2f} дБ")
        print(f"Минимум в дБ: {values_db.min():.2f} дБ")


        values_lin = 10 ** (values_db / 10)

        print(f"Максимум в линейных единицах: {values_lin.max():.4f}")


        theta_full_deg = []
        values_full_lin = []
        values_full_db = []

        for angle in range(0, 361):
            if angle <= 180:
                idx = np.argmin(np.abs(theta_deg - angle))
                theta_full_deg.append(angle)
                values_full_lin.append(values_lin[idx])
                values_full_db.append(values_db[idx])
            else:
                sym_angle = 360 - angle
                idx = np.argmin(np.abs(theta_deg - sym_angle))
                theta_full_deg.append(angle)
                values_full_lin.append(values_lin[idx])
                values_full_db.append(values_db[idx])

        theta_rad = np.deg2rad(theta_full_deg)
        values_full_lin = np.array(values_full_lin)
        values_full_db = np.array(values_full_db)

        print(f"Создан полный круг: {len(theta_full_deg)} точек (0-360°)")

        return theta_rad, values_full_lin, values_full_db

    def plot_all(self, theta_a, D_a, theta_s, D_s_lin, D_s_db):

        eps = 1e-12
        D_a_db = 10 * np.log10(np.maximum(D_a, eps))

        if len(theta_s) == 0:
            print("\nНет данных моделирования, строю только аналитику")
            theta_s = theta_a
            D_s_lin = D_a
            D_s_db = D_a_db


        max_analytic = np.max(D_a)
        max_analytic_db = 10 * np.log10(max_analytic)
        max_simulation_lin = np.max(D_s_lin)
        max_simulation_db = np.max(D_s_db)

        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 60)
        print(f"Аналитический расчет:")
        print(f"  Dmax = {max_analytic:.4f} ({max_analytic_db:.2f} дБ)")
        print(f"CST моделирование:")
        print(f"  Dmax = {max_simulation_lin:.4f} ({max_simulation_db:.2f} дБ)")
        print(f"Разница:")
        print(f"  абсолютная: {abs(max_analytic - max_simulation_lin):.4f}")
        print(f"  относительная: {abs(max_analytic - max_simulation_lin) / max_simulation_lin * 100:.2f}%")
        print(f"  в дБ: {abs(max_analytic_db - max_simulation_db):.2f} дБ")
        print("=" * 60)

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Диаграмма направленности симметричного вибратора (Вариант 20)\n"
                     f"f = {self.f / 1e9} ГГц, 2l/λ = {self.two_l_over_lambda}",
                     fontsize=14, fontweight='bold')


        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(np.rad2deg(theta_a), D_a, 'b-', linewidth=2,
                 label=f"Аналитика (max={max_analytic:.3f})")
        ax1.plot(np.rad2deg(theta_s), D_s_lin, 'r--', linewidth=1.5, alpha=0.7,
                 label=f"CST (max={max_simulation_lin:.3f})")
        ax1.set_xlabel("θ, градусы", fontsize=11)
        ax1.set_ylabel("D (линейная шкала)", fontsize=11)
        ax1.set_title("Декартова система координат (линейный масштаб)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_xlim(0, 360)
        ax1.set_xticks(np.arange(0, 361, 45))
        ax1.set_ylim(bottom=0, top=max(max_analytic, max_simulation_lin) * 1.15)


        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(np.rad2deg(theta_a), D_a_db, 'b-', linewidth=2,
                 label=f"Аналитика (max={max_analytic_db:.2f} дБ)")
        ax2.plot(np.rad2deg(theta_s), D_s_db, 'r--', linewidth=1.5, alpha=0.7,
                 label=f"CST (max={max_simulation_db:.2f} дБ)")
        ax2.set_xlabel("θ, градусы", fontsize=11)
        ax2.set_ylabel("D, дБ", fontsize=11)
        ax2.set_title("Декартова система координат (логарифмический масштаб)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_xlim(0, 360)
        ax2.set_xticks(np.arange(0, 361, 45))
        ax2.set_ylim(bottom=-40, top=max(max_analytic_db, max_simulation_db) * 1.15)


        ax3 = fig.add_subplot(2, 2, 3, projection="polar")
        ax3.plot(theta_a, D_a, 'b-', linewidth=2, label="Аналитика")
        ax3.plot(theta_s, D_s_lin, 'r--', linewidth=1.5, alpha=0.7, label="CST")
        ax3.set_title("Полярная система координат (линейный масштаб)", va="bottom", fontsize=12)
        ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax3.set_theta_zero_location("N")  # 0° наверху
        ax3.set_theta_direction(-1)  # по часовой стрелке
        ax3.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax3.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
        ax3.grid(True, alpha=0.3)
        ax3.set_rmax(max(max_analytic, max_simulation_lin) * 1.15)


        ax4 = fig.add_subplot(2, 2, 4, projection="polar")

        offset = 40
        ax4.plot(theta_a, D_a_db + offset, 'b-', linewidth=2, label="Аналитика")
        ax4.plot(theta_s, D_s_db + offset, 'r--', linewidth=1.5, alpha=0.7, label="CST")
        ax4.set_title("Полярная система координат (логарифмический масштаб)", va="bottom", fontsize=12)
        ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax4.set_theta_zero_location("N")
        ax4.set_theta_direction(-1)
        ax4.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax4.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
        ax4.grid(True, alpha=0.3)


        ticks = ax4.get_yticks()
        ax4.set_yticklabels([f'{int(t - offset)}' for t in ticks])
        ax4.set_ylim(offset - 40, offset + max(max_analytic_db, max_simulation_db) * 1.15)

        plt.tight_layout()
        plt.savefig('DN_dipole_variant20_final.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\nГрафик сохранен в файл: DN_dipole_variant20_final.png")


def main():

    path_to_file = "123.txt"

    print("=" * 60)
    print("ДИАГРАММА НАПРАВЛЕННОСТИ СИММЕТРИЧНОГО ВИБРАТОРА")
    print("Вариант 20: f = 10.5 ГГц, 2l/λ = 0.9")
    print("=" * 60)


    current_dir = os.getcwd()
    print(f"\nТекущая рабочая папка: {current_dir}")
    print(f"Проверка файла {path_to_file}: {os.path.exists(path_to_file)}")

    if not os.path.exists(path_to_file):
        print(f"\nОШИБКА: Файл {path_to_file} не найден!")
        print("Убедитесь, что файл находится в папке:", current_dir)
        print("\nДоступные txt-файлы в текущей папке:")
        txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
        if txt_files:
            for f in txt_files:
                print(f"  - {f}")
        else:
            print("  (нет txt-файлов)")
        return


    dp = DipolePattern(freq_ghz=10.5, two_l_over_lambda=0.9, data_file=path_to_file)


    theta_a, D_a, Dmax = dp.compute_analytic_D()


    theta_s, D_s_lin, D_s_db = dp.read_simulation()


    dp.plot_all(theta_a, D_a, theta_s, D_s_lin, D_s_db)


if __name__ == "__main__":
    main()
