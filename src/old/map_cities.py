import os
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import itertools
import pandas as pd
import inequalipy as ineq
import geopandas as gpd
from shapely.geometry import Point 
from shapely import wkt
import logging
import folium

file_path = './data/'

city_name='26'

df_world = gpd.read_file("./data/cb_2020_us_bg_500k.shp")
print(df_world)

destinations_df = pd.read_csv(file_path + city_name + '-destinations.csv')

print(destinations_df)

#open_optimal_le=[3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 60371041241, 60371042031, 60371065202, 60371066411, 60371211012, 60371344231, 60372014023, 60372071032, 60372430011, 60372611022, 60372622002, 60372624002, 60372933041, 60379800311]
open_optimal_le=[14447, 14448, 14449, 14450, 14451, 14452, 14453, 14454, 14455, 14456, 14457, 14458, 14459, 14460, 14461, 14462, 14463, 14464, 14465, 14466, 14467, 14468, 14469, 14470, 14471, 14472, 14473, 14474, 14475, 14476, 14477, 14478, 14479, 14480, 14481, 14482, 14483, 14484, 14485, 14486, 14487, 14488, 14489, 14490, 14491, 14492, 14493, 14494, 14495, 14496, 14497, 14498, 14499, 14500, 14501, 220710017581]
print(type(open_optimal_le[1]))
#open_optimal_le = open_optimal_le.strip('][').split(', ')

#open_optimal_le = list(map(int, open_optimal_le))

dests_le = destinations_df.loc[(destinations_df['id_dest']).isin(open_optimal_le)]

print(dests_le)

geometry = [Point(xy) for xy in zip(dests_le['lon'], dests_le['lat'])]
print(geometry)
dests_le = gpd.GeoDataFrame(dests_le, geometry=geometry,crs='epsg:4326')

print(dests_le)
existing_le = dests_le[dests_le['dest_type']=='supermarket']
print(existing_le)
new_le = dests_le[dests_le['dest_type']=='bg_centroid']
print(new_le)

dpi = 300  # set the DPI for the figure
figsize = (16, 9)  # set the figure size (you can adjust this as needed)

fig, citymap = plt.subplots(figsize=figsize, dpi=dpi)

df_world["geometry"].plot(ax=citymap, color='gold', edgecolor='goldenrod', linewidth=1)
existing_le.plot(ax=citymap, marker='o',color='black',markersize=10,label="Existing Supermarkets")
new_le.plot(ax=citymap, marker='^', color='olivedrab', markersize=100, label="Kolm Pollak LE New Stores")

citymap.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2) 

minx, miny, maxx, maxy = dests_le.total_bounds
xdif=.1*(maxx-minx)
ydif=.1*(maxy-miny)
citymap.set_xlim(minx-xdif, maxx+xdif)
citymap.set_ylim(miny-ydif, maxy+ydif)

fig.savefig('map_new_orleans.png',dpi=dpi)
plt.close(fig)


# Create a Map instance
m = folium.Map(location=[minx, miny], zoom_start=10, control_scale=True)

# Assuming existing_le and new_le are GeoDataFrames with Points and have columns 'Latitude' and 'Longitude'
for idx, row in existing_le.iterrows():
    folium.Marker([row['lat'], row['lon']], popup='Existing Store', icon=folium.Icon(color='blue')).add_to(m)

for idx, row in new_le.iterrows():
    folium.Marker([row['lat'], row['lon']], popup='New Store', icon=folium.Icon(color='green')).add_to(m)

# Save it as html
m.save('map_new_orleans.html')
