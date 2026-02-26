import sbt.*

object DependencyVersions {
  val scala2p13Version = "2.13.16"
  val scala3Version    = "3.6.4"

  val bengalStmVersion           = "0.11.0"
  val bigMathVersion             = "2.3.2"
  val catsEffectVersion          = "3.6.3"
  val catsEffectTestingVersion   = "1.7.0"
  val scalaTestVersion           = "3.2.19"
  val ejmlVersion                = "0.44.0"
  val parallelCollectionsVersion = "1.2.0"
  val smileVersion               = "3.1.1"
}

object Dependencies {
  import DependencyVersions.*

  private val bengalStm: ModuleID =
    "ai.entrolution" %% "bengal-stm" % bengalStmVersion

  private val bigMath: ModuleID =
    "ch.obermuhlner" % "big-math" % bigMathVersion

  private val catsEffect: ModuleID =
    "org.typelevel" %% "cats-effect" % catsEffectVersion

  private val catsEffectTesting: ModuleID =
    "org.typelevel" %% "cats-effect-testing-scalatest" % catsEffectTestingVersion % "test"

  private val scalaTest: ModuleID =
    "org.scalatest" %% "scalatest" % scalaTestVersion % "test"

  // Only DMatrixRMaj, CommonOps_DDRM, and LinearSolverFactory_DDRM are used
  private val ejml: ModuleID =
    "org.ejml" % "ejml-ddense" % ejmlVersion

  // Used by QuadratureIntegrator (SLQ parallel quadrature) and UniformDistribution (parallel sampling)
  private val parallelCollections: ModuleID =
    "org.scala-lang.modules" %% "scala-parallel-collections" % parallelCollectionsVersion

  private val smile: ModuleID =
    "com.github.haifengl" %% "smile-scala" % smileVersion

  val thylacine: Seq[ModuleID] =
    Seq(
      bengalStm,
      bigMath,
      catsEffect,
      catsEffectTesting,
      scalaTest,
      ejml,
      parallelCollections,
      smile
    )
}
