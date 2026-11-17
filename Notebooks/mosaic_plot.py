# %%
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
# import pyart
import re
from datetime import datetime, timedelta
import sys
import joblib
sys.path.append('/home/emanuel.alfie/datosmunin/python/Scripts')
import FUNCIONES as funciones # <--- llamo a la funcion de las funciones que se encuentra en otro directorio
from FUNCIONES import ref_cmap

# Cargamos los límites de países y provincias para poder graficarlas en los mapas
states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',edgecolor='black')

countries = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='10m',
        facecolor='none',edgecolor='black')

cart_proj = joblib.load('/home/emanuel.alfie/datosmunin/python/Archivos_variables/cart_proj.pkl')

horas_supercelda = ['20240319140000', '20240319170000', '20240319200000', '20240319230500']
horas_frente     = ['20240320140000', '20240320170000', '20240320200000', '20240320230000']

# Concateno las horas 
horas = horas_supercelda + horas_frente

path_mosaico_lats_lons = f'/home/emanuel.alfie/datosmunin/python/Archivos_variables/mosaico_data/mdbz_mosaic.20240319200000.npz'
mosaico_lats_lons = np.load(path_mosaico_lats_lons)
lats = mosaico_lats_lons["lat"]
lons = mosaico_lats_lons["lon"]

# Le digo el color que quiero que ponga a los Nans (fuera de los radares)
ref_cmap.set_bad(color='dimgray')

letras_paneles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

fig, ax = plt.subplots(2,4,figsize=(16,11), subplot_kw={'projection': cart_proj}, dpi=200)
ax = ax.flatten()

for i, hora in enumerate(horas):
        
        path_mosaico = f'/home/emanuel.alfie/datosmunin/python/Archivos_variables/mosaico_data/mdbz_mosaic.{hora}.npz'
        mosaico_nc = np.load(path_mosaico)
        mosaico = mosaico_nc["MDBZ"]

        mosaico_cf = ax[i].pcolormesh(lons, lats, mosaico, cmap=ref_cmap, 
                                vmin=0, vmax=70, transform=ccrs.PlateCarree())
        
        titulo_datetime = datetime.strptime(horas[i], '%Y%m%d%H%M%S')
        ax[i].set_title(titulo_datetime.strftime('%d/%m a las %H:00 UTC'), 
                        fontsize=16, fontweight= 'bold')
        ax[i].set_xlabel("Longitud")
        ax[i].set_ylabel("Latitud")

        gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0, linestyle='-')
        gl.top_labels = False 
        gl.right_labels = False
        gl.bottom_labels = False
        ax[i].add_feature(countries, linewidth=1)
        ax[i].add_feature(states_provinces, linewidth=1)
        ax[i].set_extent([lons.min()+1, lons.max()-1, lats.min()+0.5, -28], crs=ccrs.PlateCarree())
        ax[i].text(0.055, 0.93, letras_paneles[i], fontsize=20, fontweight='bold', transform=ax[i].transAxes)

# --- AGREGAR BARRA DE COLORES A LA DERECHA ---
plt.tight_layout(rect=[0, 0.08, 1, 1])

# [x_inicio, y_inicio, ancho, alto]
cax1 = fig.add_axes([0.25, 0.07, 0.50, 0.02]) 
cb1 = fig.colorbar(mosaico_cf, cax=cax1, 
                   ticks = np.arange(0, 71, 5),
                   orientation='horizontal')
cb1.set_label('MDBZ [dBZ]', fontsize=15)
cb1.ax.tick_params(labelsize=12)


# plt.savefig('mdbz_mosaico_caso_estudio.png', dpi=200)
plt.show()


# plt.savefig(f'/home/emanuel.alfie/datosmunin/python/Figuras/Mosaico/mdbz_mosaico_{fechas[i]}.png')