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


def pedir_funcion():
    opc = input("¿Deseas ver el manual de sintaxis primero? (s/n): ")
    if opc.lower() == "s":
        manual()

    x = Symbol("x")
    while True:
        fn_str = input("\nIngresa la función f(x): ").strip()
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


def calcular_iteraciones_teoricas(a, b, TOL):
    """Calcula n >= log2((b - a) / TOL) redondeado hacia arriba."""
    return int(np.ceil(np.log((b - a) / TOL) / np.log(2)))


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
            "f(c) o FP": FP,
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

    n_teorico = calcular_iteraciones_teoricas(a, b, TOL)
    print(f"Iteraciones teóricas necesarias: {n_teorico}")

    df_resultado = biseccion(fn, a, b, No, TOL)

    if df_resultado is not None:
        print("\n TABLA DE ITERACIONES:")
        print(df_resultado.to_string(index=False))

    input("Presiona ENTER para volver al menú principal...")    


def punto_fijo():
    datos = {
            "Iteración": [1, 2],
            "a": [0.0, 0.5],
            "b": [1.0, 1.0],
            "c": [0.5, 0.75],
            "f(c)": [-0.5, 0.125],
        }
    return pd.DataFrame(datos)

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
            input("Pendiente [Presiona ENTER para volver al menú]")
        elif opc == "3":
            input("Pendiente [Presiona ENTER para volver al menú]")
        elif opc == "4":
            print("¡Hasta luego!")
            break


if __name__ == "__main__":
    menu_inicio()