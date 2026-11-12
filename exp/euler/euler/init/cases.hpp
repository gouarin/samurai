// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Each header registers its own test cases, so adding a case means writing one
// header and adding one #include below -- there is no list to keep in sync. The
// dimensions a case is available in are stated by the macro at the bottom of its
// own header.

#include "advected_pulse.hpp"
#include "blast_periodic.hpp"
#include "closed_box.hpp"
#include "double_mach.hpp"
#include "double_rarefaction.hpp"
#include "free_stream.hpp"
#include "isentropic_vortex.hpp"
#include "kelvin_helmholtz.hpp"
#include "lax_liu.hpp"
#include "sedov_blast.hpp"
#include "sod.hpp"
#include "triple_point.hpp"
