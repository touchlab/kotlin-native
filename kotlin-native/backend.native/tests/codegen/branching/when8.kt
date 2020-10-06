/*
 * Copyright 2010-2018 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

package codegen.branching.when8

import kotlin.test.*

@Test fun runTest() {
    when (true) {
        true -> println("true")
        false -> println("false")
    }
}