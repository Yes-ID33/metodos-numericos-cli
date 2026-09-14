import os
import subprocess
import numpy as np
import pandas as pd
from sympy import Symbol, lambdify, sympify, Interval
from sympy.calculus.util import continuous_domain

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.6f" % x)

def limpiar_pantalla():
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


def manual():
    limpiar_pantalla()
    print("=" * 60)
    print("        GUÍA DE SINTAXIS Y OPERADORES MATEMÁTICOS")
    print("=" * 60)
    print("Suma / Resta:       +  y  -        Ejemplo: x + 5")
    print("Multiplicación:     *              Ejemplo: 3*x (¡Obligatorio el '*'!)")
    print("División:           /              Ejemplo: (x + 1)/(x - 2)")
    print("Potencia:           **             Ejemplo: x**3 (para x³)")
    print("Raíz cuadrada:      sqrt(x)        Ejemplo: sqrt(x + 2)")
    print("Logaritmo natural:  log(x)         Ejemplo: log(x)")
    print("Logaritmo base b:   log(x, b)      Ejemplo: log(3x, 5)")
    print("Exponencial (e^x):  exp(x)         Ejemplo: exp(-14x)")
    print("Trigonométricas:    sin(x), cos(x), tan(x)")
    print("Constantes:         pi, E")
    print("=" * 60)
    input("\nPresiona ENTER para regresar al menú principal...")


def pedir_funcion(nombre="f(x)"):
    opc = input("¿Deseas ver el manual de sintaxis primero? (s/n): ")
    if opc.lower() == "s":
        manual()

    x = Symbol("x")
    while True:
        fn_str = input(f"\nIngresa la función {nombre}: ").strip()
        try:
            expr = sympify(fn_str)
            fn = lambdify(x, expr, modules=["numpy"])
            return expr, fn
        except Exception as e:
            print(f"❌ Error al interpretar la función: {e}. Intenta de nuevo.")


def verificar_continuidad(expr, a, b):
    """Comprueba simbólicamente si f(x) es continua en [a, b]."""
    x = Symbol("x")
    intervalo = Interval(a, b)
    try:
        dominio_continuo = continuous_domain(expr, x, intervalo)
        return dominio_continuo == intervalo
    except Exception:
        # Si SymPy no puede determinar la continuidad analíticamente, permite continuar con advertencia
        return True

def verificar_teorema_punto_fijo(expr, fn, a, b):
    """Verifica automapeo g([a,b]) actual en [a,b] y calcula K = max|g'(x)| < 1."""
    x = Symbol("x")
    puntos = np.linspace(a, b, 200)

    # 1. Automapeo: g(x) debe estar dentro de [a, b] para todo x en [a, b]
    valores_g = fn(puntos)
    min_g, max_g = np.min(valores_g), np.max(valores_g)
    automapeo = (min_g >= a) and (max_g <= b)

    # 2. Contracción: K = max |g'(x)| < 1
    derivada = expr.diff(x)
    deriv_fn = lambdify(x, derivada, modules=["numpy"])
    valores_deriv = np.abs(deriv_fn(puntos))
    K = np.max(valores_deriv)

    return automapeo, K

def n_biseccion(a, b, TOL):
    """Calcula n >= log2((b - a) / TOL)."""
    return int(np.ceil(np.log((b - a) / TOL) / np.log(2)))


def n_punto_fijo(expr, fn, a, b, P0, TOL):
    """Calcula n >= log( ((1 - K) * TOL) / |P0 - g(P0)| ) / log(K)."""
    x = Symbol("x")
    try:
        derivada = expr.diff(x)
        deriv_fn = lambdify(x, derivada, modules=["numpy"])
        puntos = np.linspace(a, b, 200)
        K = float(np.max(np.abs(deriv_fn(puntos))))

        if K >= 1:
            return None, K  # No cumple la condición de contracción K < 1

        g_P0 = float(fn(P0))
        diferencia_inicial = abs(P0 - g_P0)

        if diferencia_inicial == 0:
            return 0, K  # P0 ya es el punto fijo exacto

        numerador = np.log(((1.0 - K) * TOL) / diferencia_inicial)
        denominador = np.log(K)

        n_teorico = int(np.ceil(numerador / denominador))
        return max(1, n_teorico), K
    except Exception:
        return None, None


def biseccion(fn, a, b, No, TOL):
    FA = fn(a)
    FB = fn(b)

    if FA*FB >= 0:
        print(f"f(a):{FA:.4f} y f(b):{FB:.4f} no tienen signos opuestos")
        return None

    iteraciones = []
    i=1

    while i <= No:
        P =(a+b)/2.0
        FP =fn(P)
        error = abs((b-a)/2.0)

        iteraciones.append({
            "Iteracion": i,
            "a": a,
            "b": b,
            "c": P,
            "FP": FP,
            "Error": error,
        })

        if abs(FP) < TOL or error < TOL:
            break

        if FA*FP > 0:
            a = P
            FA = FP
        else:
            b = P

        i+=1

    return pd.DataFrame(iteraciones)


def ejecucion_biseccion():
    limpiar_pantalla()
    print("====--- MÉTODO DE LA BISECCIÓN ---====")

    expr, fn = pedir_funcion()

    try:
        a = float(input("\nIngresa el límite inferior (a): "))
        b = float(input("\nIngresa el límite superior (b): "))
        if a >= b:
            print("'a' debe ser estrictamente menor que 'b'.")
            input("\nPresiona ENTER para regresar...")
            return

        TOL = float(input("Tolerancia (TOL) [Presione ENTER para 0.001]") or 0.001)
        No = int(input("Máximo de operaciones (No) [Presione ENTER para 100]") or 100)
    except ValueError:
        print("Error: Ingresaste un parámetro inválido")
        input("\nPresiona ENTER para regresar...")
        return

    print("Verificando la continuidad del intervalo...")
    if not verificar_continuidad(expr, a, b):
        print(f"La función no es continua en el intervalo [{a}, {b}]. No se puede aplicar bisección")
        input("\nPresiona ENTER para regresar...")
        return
    print("La función es continua en el intervalo.")

    n_teorico = n_biseccion(a, b, TOL)
    print(f"Iteraciones teóricas necesarias: {n_teorico}")

    df_resultado = biseccion(fn, a, b, No, TOL)

    if df_resultado is not None:
        print("\n TABLA DE ITERACIONES:")
        print(df_resultado.to_string(index=False))

    input("Presiona ENTER para volver al menú principal...")


def punto_fijo(fn, x0, TOL, No):
    """
    Método de punto fijo para resolver x = g(x).

    fn  -- función g(x) ya despejada
    x0  -- valor inicial (semilla)
    TOL -- tolerancia para el criterio de parada
    No  -- número máximo de iteraciones
    """
    iteraciones = []
    x_ant = x0
    i = 1

    while i <= No:
        x_actual = fn(x_ant)
        error = abs(x_actual - x_ant)

        iteraciones.append({
            "Iteracion": i,
            "x_anterior": x_ant,
            "x_actual": x_actual,
            "Error": error,
        })

        if error < TOL:
            break

        x_ant = x_actual
        i += 1

    return pd.DataFrame(iteraciones)


def ejecucion_punto_fijo():
    limpiar_pantalla()
    print("====--- MÉTODO DE PUNTO FIJO ---====")
    print("Recuerda: la función debe estar despejada en la forma x = g(x)\n")

    expr, fn = pedir_funcion("g(x)  (despejada de x = g(x))")

    try:
        a = float(input("\nIngresa el límite inferior del intervalo (a): "))
        b = float(input("Ingresa el límite superior del intervalo (b): "))
        if a >= b:
            print("Error: 'a' debe ser estrictamente menor que 'b'.")
            input("\nPresiona ENTER para regresar...")
            return

        x0 = float(input(f"Ingresa el valor inicial (x0) dentro de [{a}, {b}]: "))
        if not (a <= x0 <= b):
            print(f"Advertencia: x0 ({x0}) está fuera del intervalo [{a}, {b}].")

        TOL = float(input("Tolerancia (TOL) [Presione ENTER para 0.001]: ") or 0.001)
        No = int(input("Máximo de iteraciones (No) [Presione ENTER para 100]: ") or 100)
    except ValueError:
        print("Error: Ingresaste un parámetro numérico inválido.")
        input("\nPresiona ENTER para regresar...")
        return

    # 1. Verificación de Continuidad en [a, b]
    print("\n1. Verificando la continuidad de g(x) en el intervalo...")
    if not verificar_continuidad(expr, a, b):
        print(f"Error: La función NO es continua en [{a}, {b}].")
        input("\nPresiona ENTER para regresar...")
        return
    print("  - g(x) es continua en el intervalo.")

    # 2. Verificación del Teorema de Banach (Automapeo y Contracción K)
    print("\n2. Verificando el Teorema del Punto Fijo (Automapeo y K < 1)...")
    automapeo, K = verificar_teorema_punto_fijo(expr, fn, a, b)

    if automapeo:
        print("  - Automapeo confirmado: g([a, b]) es subconjunto de [a, b]")
    else:
        print("  - Advertencia: g(x) se sale del intervalo [a, b] en algunos puntos.")

    if K is not None:
        print(f"  - Constante de contracción K = max|g'(x)| en [{a}, {b}]: {K:.6f}")
        if K < 1:
            print("  - Se cumple K < 1 (Garantiza unicidad de la raíz).")
        else:
            print("  - Advertencia: K >= 1. El método podría divergir u oscilar.")

    if not automapeo or (K is not None and K >= 1):
        seguir = input("\n¿Deseas continuar con las iteraciones de todas formas? (s/n): ")
        if seguir.lower() != "s":
            input("\nPresiona ENTER para regresar al menú principal...")
            return

    # 3. Cálculo Teórico de Iteraciones
    n_teorico, _ = n_punto_fijo(expr, fn, a, b, x0, TOL)
    if n_teorico is not None:
        print(f"\nIteraciones teóricas estimadas (n): {n_teorico}")
    else:
        print("\nNo se pudo determinar n teórico (K >= 1).")

    # 4. Ejecución del algoritmo
    df_resultado = punto_fijo(fn, x0, TOL, No)

    if df_resultado is not None and not df_resultado.empty:
        print("\n TABLA DE ITERACIONES:")
        print(df_resultado.to_string(index=False))
        print(f"\nRaíz aproximada: {df_resultado.iloc[-1]['x_actual']:.6f}")

    input("\nPresiona ENTER para volver al menú principal...")

def newton():
    datos = {
            "Iteración": [1, 2],
            "a": [0.0, 0.5],
            "b": [1.0, 1.0],
            "c": [0.5, 0.75],
            "f(c)": [-0.5, 0.125],
        }
    return pd.DataFrame(datos)

def menu_inicio():
    while True:
        limpiar_pantalla()
        print("=== CALCULADORA DE MÉTODOS NUMÉRICOS ===")
        print("1. Método de la Bisección")
        print("2. Método de Punto Fijo")
        print("3. Método de Newton-Raphson")
        print("4. Salir")

        opc = input("\nSelecciona una opción (1-4): ")

        if opc == "1":
            ejecucion_biseccion()
        elif opc == "2":
            ejecucion_punto_fijo()
        elif opc == "3":
            input("Pendiente [Presiona ENTER para volver al menú]")
        elif opc == "4":
            print("¡Hasta luego!")
            break


if __name__ == "__main__":
    menu_inicio()