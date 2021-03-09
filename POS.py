############### importing necessary packages for the businesses ################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ChiantiPy.core as ch
from scipy.linalg import lu
import math
from math import log10, floor, ceil
import os

############################# User input section ###############################
element = "Ca"  # the element you want to pull from NIST
ion = "II"  # the ion you want to pull from NIST
mass = 1e-3  # solar masses
mass_number = 40  #  Sr = 88, Ca = 40
# lower case atomic symbol + ion (starting at 1 for neutral, e.g. CaII-->ca_2)
chianti_string = "ca_2"  # the species you want to pull from Chianti
no_levels = 5  #  define the number of levels of interest
temps_list = [2000]#[2000, 3500, 5000]  # K - temperature
eDens_list = [1e9]#[4.05e7, 2.77e7, 1.98e7, 1.46e7]  #  cm-3 - electron density
######## End of user input section - sit back and let the magic happen #########

############### Defining variables to be used throughout code ##################
solar_mass = 1.98847e30  # kg
boltzmann_constant = 1.38064852e-23  # J/K
nucleon_mass = 1.6726219e-27  # kg - actually proton mass
speed_of_light = 2.99792458e8  # m/s
elementary_charge = 1.60217662e-19  # Coulombs, C
planck_constant = 6.62607004e-34  #  J.s
################################################################################


def _mkdir(newdir):
    """this function is for creating the required directory for the output
    files"""
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError(
            "a file with the same name as the desired "
            "dir, '%s', already exists." % newdir
        )
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        # print "_mkdir %s" % repr(newdir)
        if tail:
            os.mkdir(newdir)
    return newdir


def read_NIST_level_info_fn(element, ion):
    """ we want to get level and line info from NIST
        we want the info for the first n levels, specified by the user
    """
    levels_url = (
        "https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&"
        f"spectrum={element}++{ion}&units=1&upper_limit=&parity_limit"
        "=both&conf_limit=All&conf_limit_begin=&conf_limit_end=&term_"
        "limit=All&term_limit_begin=&term_limit_end=&J_limit=&format=2"
        "&output=0&page_size=15&multiplet_ordered=0&conf_out=on&term_"
        "out=on&level_out=on&unc_out=on&j_out=on&g_out=on&biblio=on&"
        "temp=&submit=Retrieve+Data"
    )
    NIST_level_info = pd.read_csv(
        levels_url,
        sep=",",
        usecols=(3, 4),
        names=["g", "NIST_level(eV)"],
        header=0,
    )
    # print(NIST_level_info)
    # remove all silly characters from the df
    NIST_level_info.replace(
        regex=True,
        inplace=True,
        to_replace=r"[^0-9.\-.^A-Z.^a-z.\/.\*]",
        value=r"",
    )
    # print(NIST_level_info)

    return NIST_level_info


def read_NIST_lines_info_fn(element, ion):
    """ we want the lines info for all lines between levels 1-5 """
    lines_url = (
        "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra="
        f"{element}+{ion}&limits_type=0&low_w=&upp_w=&unit=0&de=0&"
        "format=2&line_out=0&remove_js=on&en_unit=1&output=0&bibrefs=1&"
        "page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out="
        "0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A"
        "_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_"
        "accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out="
        "on&g_out=on&submit=Retrieve+Data"
    )
    NIST_lines_info = pd.read_csv(lines_url, sep=",",)
    # print(NIST_lines_info)
    # only keep the important columns
    NIST_lines_info = NIST_lines_info[
        ["ritz_wl_vac(A)", "Aki(s^-1)", "Ei(eV)", "Ek(eV)"]
    ]
    # print(NIST_lines_info)
    # renaming the columns to something more informative
    NIST_lines_info = NIST_lines_info.rename(
        columns={
            "ritz_wl_vac(A)": "NIST_lambda(AA)",
            "Aki(s^-1)": "NIST_A-value(s-1)",
            "Ei(eV)": "NIST_ritz_E_lower(eV)",
            "Ek(eV)": "NIST_ritz_E_upper(eV)",
        }
    )
    # remove all silly characters from the df
    NIST_lines_info.replace(
        regex=True,
        inplace=True,
        to_replace=r"[^0-9.\-.^A-Z.^a-z.\/.\*]",
        value=r"",
    )
    # print(NIST_lines_info)

    # remove all lines with no defined energy levels
    indexes = NIST_lines_info[
        NIST_lines_info["NIST_ritz_E_upper(eV)"] == ""
    ].index
    NIST_lines_info = NIST_lines_info.drop(indexes.values)
    NIST_lines_info = NIST_lines_info.reset_index(drop=True)
    # pretty sure this is duplicate code but just to be sure to be sure...
    indexes = NIST_lines_info[
        NIST_lines_info["NIST_ritz_E_lower(eV)"] == ""
    ].index
    NIST_lines_info = NIST_lines_info.drop(indexes.values)
    NIST_lines_info = NIST_lines_info.reset_index(drop=True)
    # print(NIST_lines_info)

    return NIST_lines_info


def compile_NIST_info_fn(no_levels, NIST_level_info, NIST_lines_info):
    """ bundle the relevant NIST info together into 1 handy df """
    lower_lvl_indexes = []
    upper_lvl_indexes = []
    for aa in range(len(NIST_lines_info)):
        lower_lvl_energy = NIST_lines_info["NIST_ritz_E_lower(eV)"][aa]
        upper_lvl_energy = NIST_lines_info["NIST_ritz_E_upper(eV)"][aa]
        # print(lower_lvl_energy, upper_lvl_energy)
        # need to add +1 to the index as we are basing all this in Chianti
        # notation; i.e. the ground level is 1, NOT 0
        lower_lvl_index = (
            NIST_level_info[
                NIST_level_info["NIST_level(eV)"] == lower_lvl_energy
            ].index[0]
            + 1
        )
        upper_lvl_index = (
            NIST_level_info[
                NIST_level_info["NIST_level(eV)"] == upper_lvl_energy
            ].index[0]
            + 1
        )
        # print(lower_lvl_index, upper_lvl_index)
        lower_lvl_indexes.append(lower_lvl_index)
        upper_lvl_indexes.append(upper_lvl_index)

    NIST_lines_info["lower_lvl"] = lower_lvl_indexes
    NIST_lines_info["upper_lvl"] = upper_lvl_indexes

    indexes = NIST_lines_info[NIST_lines_info["upper_lvl"] > no_levels].index
    NIST_lines_info = NIST_lines_info.drop(indexes.values)
    # print(NIST_lines_info)

    return NIST_lines_info


def chianti_excitation_rates_fn(chianti_string, temps, eDens, no_levels):
    """ now we need to load in the Chianti data for CaII """
    chianti_ion = ch.ion(chianti_string, temperature=temps, eDensity=eDens)
    chianti_ion.upsilonDescale()
    lower_lvl_index = chianti_ion.Scups["lvl1"]
    upper_lvl_index = chianti_ion.Scups["lvl2"]
    exRates = chianti_ion.Upsilon["exRate"]
    dexRates = chianti_ion.Upsilon["dexRate"]

    # squeeze the ex and dex lists as they are multi-D but really are 1D
    exRates = np.squeeze(exRates)
    dexRates = np.squeeze(dexRates)

    chianti_ion_collision_df = pd.DataFrame(
        {
            "lower_lvl": lower_lvl_index,
            "upper_lvl": upper_lvl_index,
            "Chianti_exRate": exRates,
            "Chianti_dexRate": dexRates,
        }
    )
    # print(chianti_ion_collision_df)
    indexes = chianti_ion_collision_df[
        chianti_ion_collision_df["upper_lvl"] > no_levels
    ].index
    # print(indexes)
    chianti_ion_collision_df = chianti_ion_collision_df.drop(indexes.values)
    # print(chianti_ion_collision_df)

    # building df from Chianti for the wavelength, gf, A-values for transitions
    wgfa_lower_lvl = chianti_ion.Wgfa["lvl1"]
    wgfa_upper_lvl = chianti_ion.Wgfa["lvl2"]
    wgfa_lambda = chianti_ion.Wgfa["wvl"]
    wgfa_A = chianti_ion.Wgfa["avalue"]
    chianti_ion_transition_df = pd.DataFrame(
        {
            "lower_lvl": wgfa_lower_lvl,
            "upper_lvl": wgfa_upper_lvl,
            "Chianti_lambda(AA)": wgfa_lambda,
            "Chianti_A-value(s-1)": wgfa_A,
        }
    )
    # print(chianti_ion_transition_df)
    indexes = chianti_ion_transition_df[
        chianti_ion_transition_df["upper_lvl"] > no_levels
    ].index
    # print(indexes)
    chianti_ion_transition_df = chianti_ion_transition_df.drop(indexes.values)
    # print(chianti_ion_transition_df)

    chianti_ion_df = pd.merge(
        chianti_ion_collision_df,
        chianti_ion_transition_df,
        on=["lower_lvl", "upper_lvl"],
        how="left",
    )
    # print(chianti_ion_df)

    # later on, our code will crash if we don't have an entry in chianti_ion_df
    # for every downward transition, so here we append any missing ones to the
    # end of the df, and fill with zeros
    for ii in range(no_levels):  # column no./initial level
        for jj in range(no_levels):  # row no./final level
            # the indices will be 1 less than Chianti level indices
            if ii < jj:
                # checking whether the downward transition exists in
                # chianti_ion_df
                if (
                    any(
                        (chianti_ion_df.lower_lvl == ii + 1)
                        & (chianti_ion_df.upper_lvl == jj + 1)
                    )
                    is False
                ):
                    # print("we don't have a line for this transition:",
                    #       ii + 1, jj + 1)
                    chianti_ion_df.loc[len(chianti_ion_df)] = 0
                    chianti_ion_df.lower_lvl.loc[len(chianti_ion_df) - 1] = (
                        ii + 1
                    )
                    chianti_ion_df.upper_lvl.loc[len(chianti_ion_df) - 1] = (
                        jj + 1
                    )
    # print(chianti_ion_df)

    return (chianti_ion, chianti_ion_df)


def populate_rates_matrix_fn(no_levels, eDens, chianti_ion_df):
    """ now to buld the mahoosive matrix; e.g.
        start with a simple 5x5 matrix - if the first 5 levels of CaII are of
        interest, the matrix should look like:
        [
        [ 1,       1,           1,          1,          1     ],
        [R_12, -SUM{R_2j},     R_32,       R_42,       R_52   ],
        [R_13,    R_23,     -SUM{R_3j},    R_43,       R_53   ],
        [R_14,    R_24,        R_34,    -SUM{R_4j},    R_54   ],
        [R_15,    R_25,        R_35,       R_45,    -SUM{R_5j}],
        ]
        where R_ij = A_ij + n_e*C_ij
        If i < j, then no A-value exists - set to 0
        If i < j, use excitation rates
        If i > j, use de-excitation rates
    """
    Rates_matrix = np.zeros((no_levels, no_levels))

    for ii in range(no_levels):  # column no./initial level
        for jj in range(no_levels):  # row no./final level
            # the indices will be 1 less than Chianti level indices
            # print(ii+1,jj+1)
            if ii < jj:  # excitation coefficients - no A-values for excitation
                # print(chianti_ion_df['exRate'][(chianti_ion_df['lower_lvl'] == ii+1) & (chianti_ion_df['upper_lvl'] == jj+1)].values)
                # print(ii+1, jj+1)
                Rates_matrix[jj, ii] = eDens * (
                    chianti_ion_df["Scaled_exRate"][
                        (chianti_ion_df["lower_lvl"] == ii + 1)
                        & (chianti_ion_df["upper_lvl"] == jj + 1)
                    ].values[0]
                )
            elif ii > jj:  # de-excitation coefficients - need A-values here
                # as well as collisional de-excitation rates
                # to navigate the df containing line info from Chianti, we
                # cannot have ii > jj; i.e. the Chianti data does not have a
                # line for e.g. 5-->1 - instead, it has info for both
                # excitation and de-excitation for e.g. line 1-->5
                # So... need to invert the indices here to get correct values
                if (
                    #     math.isnan(
                    #         float(
                    #             chianti_ion_df["Chianti_A-value(s-1)"][
                    #                 (chianti_ion_df["lower_lvl"] == jj + 1)
                    #                 & (chianti_ion_df["upper_lvl"] == ii + 1)
                    #             ].values[0]
                    #         )
                    #     )
                    #     is True
                    # ):  # using Chianti A-values
                    math.isnan(
                        float(
                            chianti_ion_df["NIST_A-value(s-1)"][
                                (chianti_ion_df["lower_lvl"] == jj + 1)
                                & (chianti_ion_df["upper_lvl"] == ii + 1)
                            ].values[0]
                        )
                    )
                    is True
                ):  # using NIST A-values
                    A_value = 0
                    # print(ii+1, jj+1, (chianti_ion_df['A'][(chianti_ion_df['lower_lvl'] == ii2+1) & (chianti_ion_df['upper_lvl'] == jj2+1)].values[0]))
                else:
                    # A_value = float(
                    #     chianti_ion_df["Chianti_A-value(s-1)"][
                    #         (chianti_ion_df["lower_lvl"] == jj + 1)
                    #         & (chianti_ion_df["upper_lvl"] == ii + 1)
                    #     ].values[0]
                    # )  #  Chianti A-values
                    A_value = float(
                        chianti_ion_df["NIST_A-value(s-1)"][
                            (chianti_ion_df["lower_lvl"] == jj + 1)
                            & (chianti_ion_df["upper_lvl"] == ii + 1)
                        ].values[0]
                    )  #  NIST A-values
                    # A_value = 0  # can override the A-values to be zero - this will give the LTE case
                    # print(ii+1, jj+1, (chianti_ion_df['A'][(chianti_ion_df['lower_lvl'] == ii2+1) & (chianti_ion_df['upper_lvl'] == jj2+1)].values[0]))

                Rates_matrix[jj, ii] = A_value + (
                    eDens
                    * (
                        chianti_ion_df["Scaled_dexRate"][
                            (chianti_ion_df["lower_lvl"] == jj + 1)
                            & (chianti_ion_df["upper_lvl"] == ii + 1)
                        ].values[0]
                    )
                )
            elif (
                ii == jj
            ):  #  special case - the sums along the diagonal - ignore for now
                Rates_matrix[ii, jj] = 0
            else:
                print(
                    "Error! Should not have a line that causes this...",
                    ii + 1,
                    jj + 1,
                )

    # duplicating loop to now only focus on the diagonal sums
    for ii in range(no_levels):  # column no./initial level
        for jj in range(no_levels):  # row no./final level
            # the indices aa and bb will be 1 less than Chianti level indices
            # print(ii+1,jj+1)
            if ii == jj:  #  special case - sums along the diagonal
                # by this stage, the rest of the matrix has been populated,
                # so we can just pull the relevant info from that

                # this was a hard-code version for the fixed 5x5 matrix
                # Rates_matrix[jj, ii] = -1 * (
                #     Rates_matrix[0, ii]
                #     + Rates_matrix[1, ii]
                #     + Rates_matrix[2, ii]
                #     + Rates_matrix[3, ii]
                #     + Rates_matrix[4, ii]
                # )

                # this is a generalised version that works for an NxN matrix
                for aa in range(no_levels):
                    # this if statement stops the code from adding the location
                    # of the current matrix element to itself
                    if aa != ii:
                        Rates_matrix[jj, ii] += Rates_matrix[aa, ii]
                #  need to multiply by -1 at the end as these entries are -ve
                Rates_matrix[jj, ii] = Rates_matrix[jj, ii] * -1

    # overwrite the top row of the matrix to all be 1
    Rates_matrix[0, :] = 1

    return Rates_matrix


########################## Start of main body of code ##########################

for temps in temps_list:

    # we use this for the evolution of the level populations through time
    offset_bars = -1 * (len(eDens_list) - 1) / 2

    counter = 0

    for eDens in eDens_list:
        # pulling all the relevant info from NIST levels and lines
        NIST_level_info = read_NIST_level_info_fn(element, ion)
        # print(NIST_level_info)
        NIST_lines_info = read_NIST_lines_info_fn(element, ion)
        # print(NIST_lines_info)
        NIST_lines_info = compile_NIST_info_fn(
            no_levels, NIST_level_info, NIST_lines_info
        )
        # print(NIST_lines_info)

        # defining the ion in Chianti and pulling out some info on
        # (de-)excitation rates
        chianti_ion, chianti_ion_df = chianti_excitation_rates_fn(
            chianti_string, temps, eDens, no_levels
        )
        # print(chianti_ion_df)

        # merging the NIST and Chianti df's together to form one mega-df which
        # can do everything those other df's wish they could
        chianti_ion_df2 = pd.merge(
            chianti_ion_df,
            NIST_lines_info,
            on=["lower_lvl", "upper_lvl"],
            how="left",
        )
        # print(chianti_ion_df2)

        chianti_ion_df2["Scaled_exRate"] = (
            chianti_ion_df2["Chianti_exRate"]
            / np.exp(
                -1
                * (planck_constant * speed_of_light)
                / (
                    chianti_ion_df2["Chianti_lambda(AA)"]
                    * 1e-10
                    * boltzmann_constant
                    * temps
                )
            )
            * np.exp(
                -1
                * (planck_constant * speed_of_light)
                / (
                    chianti_ion_df2["NIST_lambda(AA)"].astype("float")
                    * 1e-10
                    * boltzmann_constant
                    * temps
                )
            )
        )
        for cc in range(len(chianti_ion_df2)):
            if math.isnan(chianti_ion_df2["Scaled_exRate"][cc]) is True:
                chianti_ion_df2["Scaled_exRate"][cc] = chianti_ion_df2[
                    "Chianti_exRate"
                ][cc]
        chianti_ion_df2["Scaled_dexRate"] = chianti_ion_df2[
            "Chianti_dexRate"
        ]  #  de-excitation does not change
        # print(chianti_ion_df2)

        # here we are generating the rates matrix
        Rates_matrix = populate_rates_matrix_fn(
            no_levels, eDens, chianti_ion_df2
        )

        # use the magic package to solve the equation of the form A.x = y,
        # where A = Rates_matrix, x = level populations, and y = y_array
        # (see literally the next 3 lines)
        y_array = np.zeros(no_levels)
        y_array[0] = 1
        level_pops = np.linalg.solve(Rates_matrix, y_array)
        # print(level_pops)

        #  for comparison, generate the level populations in Chianti
        chianti_ion.populate()
        # print(chianti_ion.Population['population'])
        chianti_ion_level_pops = np.squeeze(
            chianti_ion.Population["population"]
        )
        chianti_ion_level_pops = chianti_ion_level_pops[:no_levels]

        # now calculate the LTE level populations for comparison too
        LTE_level_pops = np.zeros(
            no_levels
        )  #  empty array to store the LTE level populations

        # don't need to know partition function value as all
        # level populations will be normalised anyway so...
        Z = 1
        for aa in range(no_levels):
            g_weight = NIST_level_info["g"][aa]
            lvl_energy = (
                float(NIST_level_info["NIST_level(eV)"][aa]) * elementary_charge
            )
            LTE_level_pops[aa] = (g_weight / Z) * np.exp(
                (-1 * lvl_energy) / (boltzmann_constant * temps)
            )

        LTE_level_pops = LTE_level_pops / np.sum(LTE_level_pops)

        # define levels to assign level populations to
        # do them this way to stick with Chianti notation, i.e. ground=1, not 0
        levels = np.arange(1, no_levels + 1)
        # print(levels)

        # if length is 1, then plot Lte vs. Chianti vs POS level populations
        if len(temps_list) == 1:
            # bar chart representation of level populations
            width = 0.2
            plt.bar(
                levels - width,
                LTE_level_pops,
                width,
                label="LTE",
                color="cornflowerblue",
            )
            plt.bar(
                levels,
                chianti_ion_level_pops,
                width,
                label="Chianti",
                color="darkred",
            )
            plt.bar(
                levels + width,
                level_pops,
                width,
                label="POS",
                color="darkorange",
            )

            plt.yscale("log")
            plt.legend(loc="best")
            plt.xlabel("Level (Chianti Notation)")
            plt.ylabel("Relative Population")
            plt.title(f"{element}{ion} @ {temps:.1E}K")
            # determining the minimum y-axis value
            abs_min = min(
                min(LTE_level_pops), min(chianti_ion_level_pops), min(level_pops)
            )
            abs_min = round(abs_min, -int(ceil(log10(abs(abs_min))))) / 10
            plt.ylim(
                [abs_min, 2]
            )  # max value always 1 as the level populations have been normalised

            # ensures that the axis labels are always correct
            xlabels = np.arange(1, no_levels + 1)
            plt.xticks(xlabels, xlabels)

            # plt.show()
            plt.savefig(
                f"{_mkdir('bar_charts')}/{element}{ion}-{chianti_string}-"
                f"{temps:.1E}K_level_populations-scaled_excitations.png",
                dpi=900,
                bbox_inches="tight",
            )
            plt.close()

        # output this level population stuff to a file
        level_pops_df = pd.DataFrame(
            {
                "lvl": levels,
                "LTE_lvl_pops": LTE_level_pops,
                "Chianti_lvl_pops": chianti_ion_level_pops,
                "Custom_lvl_pops": level_pops,
            }
        )
        level_pops_df.to_csv(
            f"{_mkdir('output_csvs')}/{element}{ion}-{chianti_string}-"
            f"{temps:.1E}K_rho_e-{eDens:.2E}_level_populations-scaled_"
            "excitations.csv",
            sep=",",
            index=False,
        )
        print(level_pops_df)
        print(chianti_ion_df2)

        upper_lvl_list = []
        lower_lvl_list = []
        wavelength_list = []
        line_lum_list = []
        # now work out luminosites of these lines
        for bb in range(len(chianti_ion_df2)):
            level_pop = level_pops_df["LTE_lvl_pops"][
                level_pops_df["lvl"][
                    level_pops_df["lvl"] == chianti_ion_df2["upper_lvl"][bb]
                ].index
            ].values[0]
            # print(level_pop)
            no_upper_level = level_pop * (
                (mass * solar_mass) / (nucleon_mass * mass_number)
            )
            # print(no_upper_level)
            no_photons = (
                float(chianti_ion_df2["NIST_A-value(s-1)"][bb]) * no_upper_level
            )
            # print(no_photons)
            wavelength = float(chianti_ion_df2["NIST_lambda(AA)"][bb])
            line_lum = (
                no_photons
                * ((planck_constant * speed_of_light) / (wavelength * 1e-10))
            ) / 1e-7  #  erg/s
            # gets rid of nasty nan's - set them to zero instead
            if math.isnan(wavelength) is True:
                wavelength = 0
            if math.isnan(line_lum) is True:
                # print(chianti_ion_df2["upper_lvl"][bb], chianti_ion_df2["lower_lvl"][bb], line_lum)
                line_lum = 0
                # print(chianti_ion_df2["upper_lvl"][bb], chianti_ion_df2["lower_lvl"][bb], line_lum)
            # print(
            #     chianti_ion_df2["upper_lvl"][bb],
            #     chianti_ion_df2["lower_lvl"][bb],
            #     line_lum,
            # )
            upper_lvl_list.append(chianti_ion_df2["upper_lvl"][bb])
            lower_lvl_list.append(chianti_ion_df2["lower_lvl"][bb])
            wavelength_list.append(wavelength)
            line_lum_list.append(line_lum)

        line_lum_df = pd.DataFrame(
            {
                "upper_lvl": upper_lvl_list,
                "lower_lvl": lower_lvl_list,
                "lambda(AA)": wavelength_list,
                "line_lum(erg/s)": line_lum_list,
            }
        )
        line_lum_df.to_csv(
            f"{_mkdir('output_csvs')}/{element}{ion}-{chianti_string}-"
            f"{temps:.1E}K_rho_e-{eDens:.2E}_line_luminosities.csv",
            sep=",",
            index=False,
        )
        print(line_lum_df)

        # if more than 1 temp or eDens specified, then plot evolution of level
        # population to show how populations vary
        if len(temps_list) > 1 or len(eDens_list) > 1:
            # do some visualisation of how level population evolves
            # through time, i.e. decreasing electron density
            width = 0.1
            # plt.bar(
            #     levels - (width * offset_factor),
            #     LTE_level_pops,
            #     width,
            #     label="LTE",
            #     color="cornflowerblue",
            # )
            # plt.bar(
            #     levels, chianti_ion_level_pops, width, label="Chianti", color="darkred"
            # )
            cm_cubed_string = r"$\mathrm{cm}^{-3}$"
            plt.bar(
                level_pops_df["lvl"] + (width * offset_bars),
                level_pops_df["Custom_lvl_pops"],
                width,
                label=f"{eDens:.2E} {cm_cubed_string}",
                # color="darkorange",
            )

            for dd in range(len(level_pops_df)):
                text_string = f"{level_pops_df['Custom_lvl_pops'][dd]:.2E}"
                plt.text(
                    level_pops_df["lvl"][dd] + (width * offset_bars) - 0.05,
                    1 / (10 ** (counter + 1)),
                    text_string,
                    fontsize=8,
                )

            counter += 1

            # Make some labels
            # ax = level_pops_df['Custom_lvl_pops'].plot(kind='bar')
            # rects = ax.patches
            # labels = ["label%d" % i for i in range(len(rects))]

            # for rect, label in zip(rects, labels):
            #     height = rect.get_height()
            #     plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
            #             ha='center', va='bottom')

            offset_bars += 1

    # if more than 1 temp or eDens specified, then plot evolution of level
    # population to show how populations vary
    if len(temps_list) > 1 or len(eDens_list) > 1:
        plt.yscale("log")
        plt.legend(
            bbox_to_anchor=(0.5, 1.03),
            ncol=4,
            loc="center",
            frameon=False,
            fontsize=7,
        )
        plt.xlabel("Level (Chianti Notation)")
        plt.ylabel("Relative Population")
        # plt.title(f"{element}{ion} @ {temps:.1E}K")
        # determining the minimum y-axis value
        abs_min = min(
            min(LTE_level_pops), min(chianti_ion_level_pops), min(level_pops)
        )
        abs_min = round(abs_min, -int(ceil(log10(abs(abs_min))))) / 10
        plt.ylim(
            [abs_min, 2]
        )  # max value always 1 as the level populations have all been normalised
        # plt.show()
        plt.savefig(
            f"{_mkdir('evolution_of_level_pops')}/{element}{ion}-{chianti_string}-"
            f"{temps:.1E}K_level_populations-scaled_excitations.png",
            dpi=900,
            bbox_inches="tight",
        )
        plt.close()
