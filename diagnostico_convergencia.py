"""Entrada compativel para o diagnostico completo de convergencia."""
from threadpoolctl import threadpool_limits
from revisar_convergencia import main

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
