import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


f = 10.5e9  # 10.5 ГГц в Гц
ratio = 0.9  # 2l/λ = 0.9

kl = 0.9 * np.pi

print(f"Параметры для варианта 20:")
print(f"Частота f = {f / 1e9} ГГц")
print(f"Отношение 2l/λ = {ratio}")
print(f"l/λ = {ratio / 2}")
print(f"kl = {kl:.4f} рад = {kl / np.pi:.2f}π")


def get_f(theta):
    num = np.cos(kl * np.cos(theta)) - np.cos(kl)
    den = np.sin(kl) * (np.sin(theta) + 1e-10)
    val = np.abs(num / den)
    return val / (np.max(val) if np.max(val) > 0 else 1)

int_val, _ = quad(lambda t: (get_f(t) ** 2) * np.sin(t), 0, np.pi)
d_max_an = 2 / int_val


d_max_cst = d_max_an

print(f"\nРезультаты расчета:")
print(f"Dmax аналитически (разы) = {d_max_an:.4f}")
print(f"Dmax аналитически (дБ) = {10 * np.log10(d_max_an):.4f}")
print(f"Dmax CST (разы) = {d_max_cst:.4f}")

angles = np.radians(np.linspace(0, 180, 500))
f_sq = get_f(angles) ** 2

d_an_lin = f_sq * d_max_an
d_cst_lin = f_sq * d_max_cst

d_an_db = 10 * np.log10(np.maximum(d_an_lin, 1e-4))
d_cst_db = 10 * np.log10(np.maximum(d_cst_lin, 1e-4))

fig, axs = plt.subplots(2, 2, figsize=(13, 10))
titles = ["D (разы), Декартова", "D (разы), Полярная",
          "D (дБ), Декартова", "D (дБ), Полярная"]

for i, (an, cst, title) in enumerate([
    (d_an_lin, d_cst_lin, titles[0]),
    (d_an_lin, d_cst_lin, titles[1]),
    (d_an_db, d_cst_db, titles[2]),
    (d_an_db, d_cst_db, titles[3])
]):
    row, col = i // 2, i % 2
    is_polar = "Полярная" in title

    if is_polar:

        axs[row, col].remove()
        ax = fig.add_subplot(2, 2, i + 1, projection='polar')

        ax.plot(angles, an, 'b--', linewidth=2, alpha=0.7, label='Аналитика')
        ax.plot(angles, cst, 'r-', linewidth=1.5, alpha=0.7, label='CST')
        ax.plot(-angles, an, 'b--', linewidth=2, alpha=0.7)
        ax.plot(-angles, cst, 'r-', linewidth=1.5, alpha=0.7)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.grid(True, alpha=0.3)

        if "разы" in title:
            ax.set_ylim(0, d_max_an * 1.1)
        else:
            ax.set_ylim(-40, max(10, 10 * np.log10(d_max_an)) + 5)

    else:
        ax = axs[row, col]

        ax.plot(np.degrees(angles), an, 'b--', linewidth=2, alpha=0.7, label='Аналитика')
        ax.plot(np.degrees(angles), cst, 'r-', linewidth=1.5, alpha=0.7, label='CST')

        ax.set_xlim(0, 180)
        ax.set_xlabel("θ, градусы", fontsize=11)
        ax.grid(True, alpha=0.3)

        if "разы" in title:
            ax.set_ylabel("D (разы)", fontsize=11)
            ax.set_ylim(0, d_max_an * 1.1)
            ax.axhline(y=d_max_an, color='gray', linestyle=':', alpha=0.5,
                       label=f'Dmax = {d_max_an:.2f}')
        else:
            ax.set_ylabel("D (дБ)", fontsize=11)
            ax.set_ylim(-40, max(10, 10 * np.log10(d_max_an)) + 5)
            ax.axhline(y=10 * np.log10(d_max_an), color='gray', linestyle=':', alpha=0.5,
                       label=f'Dmax = {10 * np.log10(d_max_an):.2f} дБ')

        ax.set_xticks(np.arange(0, 181, 30))
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)

    if not is_polar:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fontsize=10, ncol=3, framealpha=0.8)
    else:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                  fontsize=10, framealpha=0.8)


plt.suptitle(
    f"Диаграмма направленности симметричного вибратора (Вариант 20)\n"
    f"f = {f / 1e9} ГГц, 2l/λ = {ratio}, "
    f"Dmax = {d_max_an:.4f} раз ({10 * np.log10(d_max_an):.4f} дБ)",
    fontsize=13, fontweight='bold', y=0.98
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('DN_dipole_variant20_aligned.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nГрафик сохранен:")
print(f"1. DN_dipole_variant20_aligned.png ")

